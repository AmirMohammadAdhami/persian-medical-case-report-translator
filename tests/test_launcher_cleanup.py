"""
Tests for the launcher's process cleanup (run_opencode_bridge.py).

The hard problem here is Windows-specific: when a process is killed with
TerminateProcess() no user-mode code runs, so signal handlers and try/finally
cannot clean up children. A job object with KILL_ON_JOB_CLOSE is the only thing
that covers that case, because the kernel closes our handle when we die.
"""

import os
import subprocess
import sys

import pytest

from run_opencode_bridge import _ChildSupervisor, _WindowsJobObject


def test_job_object_creation_is_safe_and_boolean():
    """create() must return a bool and never raise, on any platform."""
    job = _WindowsJobObject()
    result = job.create()
    assert isinstance(result, bool)
    assert job.active == result


def test_job_object_inactive_off_windows():
    """Off Windows the job is simply unavailable; the launcher falls back."""
    if os.name == "nt":
        pytest.skip("Windows-specific assertion")
    job = _WindowsJobObject()
    assert job.create() is False
    assert job.active is False


@pytest.mark.skipif(os.name != "nt", reason="Windows job objects only")
def test_child_is_assigned_to_job():
    """A live child process can be bound to the job."""
    job = _WindowsJobObject()
    if not job.create():
        pytest.skip("job object unavailable in this environment")

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert job.add(child) is True
    finally:
        child.kill()
        child.wait()


class _FakeJob:
    """Records assignment attempts so we can assert the supervisor wires it up."""

    def __init__(self):
        self.assigned = []

    def add(self, process):
        self.assigned.append(process)
        return True


def test_supervisor_hands_children_to_job():
    job = _FakeJob()
    supervisor = _ChildSupervisor(job=job)

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    supervisor.add(child)
    child.wait()

    assert job.assigned == [child]
    assert supervisor.children == [child]


def test_supervisor_works_without_job():
    """Job binding is optional; the supervisor still tracks children."""
    supervisor = _ChildSupervisor()
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    supervisor.add(child)
    child.wait()

    assert supervisor.children == [child]
    # stop_all on an already-exited process must not raise.
    supervisor.stop_all()
