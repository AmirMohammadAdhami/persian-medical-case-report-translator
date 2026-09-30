"""
Convenience launcher that starts BOTH halves of the OpenCode free-tier path:

  1. `opencode serve`  — the official client's headless server
  2. the OpenAI-compatible bridge that fronts it

Run this, then point the translator at http://127.0.0.1:<bridge-port>/v1

    python run_opencode_bridge.py

Or in one shot, without this script:

    # terminal 1
    opencode serve --port 4096

    # terminal 2
    python -m case_translator.opencode_bridge --port 4097 --model mimo-v2.6-flash-free
"""

import argparse
import os
import pathlib
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import List

DEFAULT_SERVER_PORT = 4096
DEFAULT_BRIDGE_PORT = 4097
DEFAULT_MODEL = "mimo-v2.6-flash-free"


class _WindowsJobObject:
    """
    Binds child processes to a job object that kills them when *this* process
    dies for any reason — including a hard kill.

    Why this exists: on Windows, terminating a process with TerminateProcess()
    (what Task Manager's "End task", `taskkill /F`, and most task runners do)
    runs NO user-mode code. Signal handlers never fire and `try/finally` never
    runs, so a pure-Python supervisor cannot clean up its children — they are
    orphaned with their ports still bound. `_ChildSupervisor` covers SIGINT and
    SIGTERM, but only a job object covers the hard-kill case: the kernel closes
    our job handle when we die, and that triggers KILL_ON_JOB_CLOSE.

    This is best-effort. If the job cannot be created (very old Windows, or a
    sandbox that forbids it), we fall back to signal-handling only.
    """

    # Win32 constants
    _JobObjectExtendedLimitInformation = 9
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

    def __init__(self) -> None:
        self._kernel32 = None
        self._handle = None

    def create(self) -> bool:
        if os.name != "nt":
            return False
        try:
            import ctypes
            from ctypes import wintypes

            class _BasicLimits(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class _IoCounters(ctypes.Structure):
                _fields_ = [
                    (name, ctypes.c_ulonglong)
                    for name in (
                        "ReadOperationCount",
                        "WriteOperationCount",
                        "OtherOperationCount",
                        "ReadTransferCount",
                        "WriteTransferCount",
                        "OtherTransferCount",
                    )
                ]

            class _ExtendedLimits(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", _BasicLimits),
                    ("IoInfo", _IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateJobObjectW.restype = wintypes.HANDLE
            kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
            kernel32.SetInformationJobObject.argtypes = [
                wintypes.HANDLE,
                ctypes.c_int,
                wintypes.LPVOID,
                wintypes.DWORD,
            ]
            kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                return False

            info = _ExtendedLimits()
            info.BasicLimitInformation.LimitFlags = (
                self._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            )
            ok = kernel32.SetInformationJobObject(
                handle,
                self._JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if not ok:
                kernel32.CloseHandle(handle)
                return False

            self._kernel32 = kernel32
            self._handle = handle
            return True
        except Exception:
            # Never let job-object setup break the launcher.
            return False

    def add(self, process: subprocess.Popen) -> bool:
        """Assigns a running child to the job. Returns True on success."""
        if not self._handle or process.poll() is not None:
            return False
        try:
            handle = int(process._handle)  # Popen's Win32 HANDLE
            return bool(self._kernel32.AssignProcessToJobObject(self._handle, handle))
        except Exception:
            return False

    @property
    def active(self) -> bool:
        return self._handle is not None


class _ChildSupervisor:
    """
    Tracks child processes so none are orphaned when this launcher exits.

    Without this, killing the launcher (Ctrl-C, or SIGTERM from `timeout`)
    leaves `opencode serve` and the bridge running with no owner. SIGTERM is
    especially important because it does NOT raise KeyboardInterrupt, so a
    plain try/finally would never run.

    On Windows every child is additionally bound to a job object, which covers
    the hard-kill case that no Python-level handler can observe.
    """

    def __init__(self, job: "_WindowsJobObject | None" = None) -> None:
        self.children: List[subprocess.Popen] = []
        self._installed = False
        self._job = job

    def add(self, process: subprocess.Popen) -> subprocess.Popen:
        self.children.append(process)
        if self._job is not None:
            self._job.add(process)
        return process

    def install_signal_handlers(self) -> None:
        def _handler(signum, _frame):
            self.stop_all()
            sys.exit(0)

        for signum in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
            if signum is None:
                continue
            try:
                signal.signal(signum, _handler)
                self._installed = True
            except (ValueError, OSError):
                # Signal handling is unavailable in some embedded contexts.
                pass

    def stop_all(self) -> None:
        for process in self.children:
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
        # Give them a moment, then force anything still alive.
        deadline = time.time() + 5
        for process in self.children:
            while time.time() < deadline and process.poll() is None:
                time.sleep(0.1)
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass


def resolve_opencode(argv_tail: List[str]) -> List[str]:
    """
    Builds a Popen argv for the opencode CLI that actually works on Windows.

    npm installs several wrappers on Windows: an extension-less bash shim
    (`opencode`, `#!/bin/sh`) plus `opencode.cmd` and `opencode.ps1`. Passing the
    bare string "opencode" to subprocess.Popen fails with FileNotFoundError,
    because CreateProcess does not perform PATHEXT lookup. We must pass the
    resolved path (e.g. ...\\npm\\opencode.CMD) instead.

    If only the bash shim resolves, fall back to running through the shell so
    the shim is interpreted by a shell rather than executed directly.
    """
    resolved = shutil.which("opencode")
    if not resolved:
        raise FileNotFoundError(
            "The 'opencode' CLI was not found on PATH. "
            "Install it with: npm install -g opencode-ai"
        )

    if os.name == "nt":
        suffix = pathlib.Path(resolved).suffix.lower()
        if suffix in (".exe", ".cmd", ".bat", ".com"):
            return [resolved, *argv_tail]
        # Extension-less (bash) shim: prefer a .cmd sibling if npm created one.
        for candidate in (resolved + ".cmd", resolved + ".bat", resolved + ".exe"):
            if os.path.exists(candidate):
                return [candidate, *argv_tail]
        # Last resort: let the shell interpret the shim.
        return [" ".join([shlex.quote(resolved), *argv_tail])]

    return [resolved, *argv_tail]


def wait_for_server(url: str, timeout_seconds: int = 60) -> bool:
    """Polls the opencode server health endpoint until it responds."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/global/health", timeout=5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(1.5)
    return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Start the opencode server and the OpenAI-compatible bridge together."
    )
    parser.add_argument("--server-port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--bridge-port", type=int, default=DEFAULT_BRIDGE_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--skip-server",
        action="store_true",
        help="Bridge only — use when `opencode serve` is already running",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("OPENCODE_SERVER_PASSWORD"),
        help="Password, if the opencode server is protected with basic auth",
    )
    args = parser.parse_args(argv)

    try:
        serve_argv = resolve_opencode(
            ["serve", "--port", str(args.server_port), "--hostname", "127.0.0.1"]
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    job = _WindowsJobObject()
    if job.create():
        print("[run] children bound to a job object (auto-killed even on hard exit)")
    else:
        print("[run] job object unavailable — falling back to signal handlers only")

    supervisor = _ChildSupervisor(job=job)
    supervisor.install_signal_handlers()

    server_url = f"http://127.0.0.1:{args.server_port}"
    server_process = None

    if not args.skip_server:
        # Reuse an already-running server on this port if one exists.
        if wait_for_server(server_url, timeout_seconds=2):
            print(f"[run] opencode server already running at {server_url}")
        else:
            print(f"[run] starting: opencode serve --port {args.server_port}")
            print(f"[run] resolved CLI path: {serve_argv[0]}")
            server_process = supervisor.add(
                subprocess.Popen(
                    serve_argv,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
            if not wait_for_server(server_url, timeout_seconds=60):
                print(
                    f"Error: the opencode server did not become healthy at {server_url}.\n"
                    "Try running `opencode serve` manually to see the error output.",
                    file=sys.stderr,
                )
                if server_process:
                    server_process.terminate()
                return 1
            print(f"[run] opencode server is healthy at {server_url}")

    env = dict(os.environ)
    if args.password:
        # Passed through the environment, never on the command line: argv is
        # world-readable in the process list (ps / Task Manager).
        env["OPENCODE_SERVER_PASSWORD"] = args.password

    # The bridge authenticates its clients. Reuse OPENCODE_BRIDGE_TOKEN when the
    # user set one, otherwise mint a token here and print it ready to paste into
    # .env as OPENCODE_API_KEY.
    bridge_token = os.getenv("OPENCODE_BRIDGE_TOKEN") or secrets.token_urlsafe(24)
    env["OPENCODE_BRIDGE_TOKEN"] = bridge_token

    print(f"[run] starting bridge on port {args.bridge_port} (model: {args.model})")
    print()
    print("  Point the translator at this endpoint:")
    print(f'    OPENCODE_BASE_URL="http://127.0.0.1:{args.bridge_port}/v1"')
    print(f'    OPENCODE_MODEL="{args.model}"')
    print(f'    OPENCODE_API_KEY="{bridge_token}"')
    print()
    print("  Then, in another terminal:")
    print('    python main.py "sample_case_report.pdf" --provider opencode')
    print()

    bridge = supervisor.add(
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "case_translator.opencode_bridge",
                "--port",
                str(args.bridge_port),
                "--opencode-url",
                server_url,
                "--model",
                args.model,
            ],
            env=env,
        )
    )

    try:
        bridge.wait()
    except KeyboardInterrupt:
        print("\n[run] stopping.")
    finally:
        supervisor.stop_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
