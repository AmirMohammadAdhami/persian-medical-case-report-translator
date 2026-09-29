"""
OpenCode bridge — an OpenAI-compatible local gateway backed by the official
opencode CLI.

Why this exists
---------------
The OpenCode Zen *free tier* rejects direct third-party API calls with HTTP 403.
Since 2026-09-16 the free tier requires a correctly formatted
``x-opencode-session`` header, which only the official client sets. Forging that
header is an access-control bypass, so we do not do it.

Instead this bridge runs on top of the official client's own server
(``opencode serve``). The official client performs the outbound request, so the
free tier authenticates normally and no header tricks are required.

Topology
--------
    case_translator.OpenCodeTranslator  (OpenAI SDK, chat.completions)
        -> http://127.0.0.1:<port>/v1/chat/completions     <- this bridge
            -> opencode CLI server session endpoints       <- official client
                -> OpenCode Zen free models

Usage
-----
    python -m case_translator.opencode_bridge --port 4096

Then point the project at it:

    OPENCODE_BASE_URL="http://127.0.0.1:4096/v1"
    OPENCODE_MODEL="mimo-v2.6-flash-free"
    OPENCODE_API_KEY="local"     # value is ignored by the bridge

The bridge speaks just enough OpenAI Chat Completions for this project's usage
(single system + user message, non-streaming) and translates that onto
opencode's native session API.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_OPENCODE_SERVER = "http://127.0.0.1:4096"
SESSION_TIMEOUT_SECONDS = 600


class OpenCodeClientError(RuntimeError):
    """Raised when the opencode server returns an error."""


class OpenCodeServerClient:
    """Thin client for the opencode CLI's native HTTP API."""

    def __init__(self, base_url: str, password: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.password = password

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout: int = SESSION_TIMEOUT_SECONDS,
    ) -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Content-Type", "application/json")
        if self.password:
            token = base64.b64encode(
                f"opencode:{self.password}".encode("utf-8")
            ).decode("ascii")
            request.add_header("Authorization", f"Basic {token}")

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
                return json.loads(body) if body.strip() else None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OpenCodeClientError(
                f"{method} {path} failed with HTTP {exc.code}: {detail[:400]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OpenCodeClientError(
                f"Cannot reach the opencode server at {self.base_url}. "
                f"Is `opencode serve` running? ({exc.reason})"
            ) from exc

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/global/health", timeout=10)

    def create_session(self, title: str = "case-report-translator") -> str:
        session = self._request("POST", "/session", {"title": title})
        session_id = (session or {}).get("id")
        if not session_id:
            raise OpenCodeClientError(f"Server did not return a session id: {session}")
        return session_id

    def delete_session(self, session_id: str) -> None:
        try:
            self._request("DELETE", f"/session/{session_id}", timeout=30)
        except OpenCodeClientError:
            pass  # best-effort cleanup

    def send_message(
        self,
        session_id: str,
        text: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
    ) -> str:
        """Sends a message and returns the assistant's concatenated text."""
        body: Dict[str, Any] = {
            "parts": [{"type": "text", "text": text}],
        }
        if system:
            body["system"] = system
        if model:
            provider_id, _, model_id = model.partition("/")
            if model_id:
                body["model"] = {"providerID": provider_id, "modelID": model_id}
            else:
                body["model"] = {"providerID": "opencode", "modelID": provider_id}

        result = self._request("POST", f"/session/{session_id}/message", body)
        return extract_text(result)


def extract_text(message_response: Any) -> str:
    """
    Pulls assistant text out of an opencode message response.

    Shape is { info: Message, parts: Part[] }. We collect text parts, ignoring
    reasoning/tool parts that some models emit.
    """
    if not isinstance(message_response, dict):
        return ""
    parts = message_response.get("parts") or []
    chunks: List[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in ("text", None):
            text = part.get("text")
            if text:
                chunks.append(text)
    return "\n".join(chunks).strip()


# --------------------------------------------------------------------------
# OpenAI <-> opencode translation
# --------------------------------------------------------------------------

def split_messages(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], str]:
    """
    Flattens an OpenAI messages array into (system_prompt, user_text).

    This project always sends exactly one system + one user message, but we
    handle multi-turn input defensively by labelling the roles.
    """
    system_parts: List[str] = []
    conversation: List[str] = []

    for message in messages or []:
        role = (message.get("role") or "user").lower()
        content = message.get("content")

        if isinstance(content, list):
            # Multimodal content: keep only text segments.
            content = " ".join(
                segment.get("text", "")
                for segment in content
                if isinstance(segment, dict) and segment.get("type") == "text"
            )

        content = (content or "").strip()
        if not content:
            continue

        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            conversation.append(f"Assistant: {content}")
        else:
            conversation.append(content)

    system_prompt = "\n\n".join(system_parts) or None
    user_text = "\n\n".join(conversation)
    return system_prompt, user_text


def strip_provider_prefix(model: str) -> str:
    """'opencode/mimo-v2.6-flash-free' -> 'mimo-v2.6-flash-free'."""
    model = (model or "").strip()
    if "/" in model:
        return model.split("/", 1)[1]
    return model


class BridgeHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible handler backed by the opencode CLI."""

    server_version = "opencode-bridge/1.0"
    opencode: OpenCodeServerClient
    default_model: str

    # ---- helpers ---------------------------------------------------------

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str, err_type: str = "bridge_error") -> None:
        self._send_json(status, {"error": {"message": message, "type": err_type}})

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        sys.stderr.write(f"[bridge] {self.address_string()} {fmt % args}\n")

    # ---- routes ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.default_model,
                            "object": "model",
                            "owned_by": "opencode",
                        }
                    ],
                },
            )
            return
        if self.path.rstrip("/") in ("/health", "/global/health"):
            try:
                info = self.opencode.health()
                self._send_json(200, {"healthy": True, "opencode": info})
            except OpenCodeClientError as exc:
                self._send_error(503, str(exc))
            return
        self._send_error(404, f"Unknown path: {self.path}", "not_found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._send_error(404, f"Unknown path: {self.path}", "not_found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(raw or "{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_error(400, f"Invalid JSON body: {exc}", "invalid_request_error")
            return

        messages = payload.get("messages") or []
        model = strip_provider_prefix(payload.get("model") or self.default_model)
        system_prompt, user_text = split_messages(messages)

        if not user_text:
            self._send_error(400, "No user message content found.", "invalid_request_error")
            return

        if payload.get("stream"):
            # Streaming is not needed by this project. Fail loudly rather than
            # silently returning a non-streamed body the client cannot parse.
            self._send_error(
                400,
                "Streaming is not supported by the opencode bridge. Send stream=false.",
                "invalid_request_error",
            )
            return

        try:
            content = self._translate(model, system_prompt, user_text)
        except OpenCodeClientError as exc:
            self._send_error(502, str(exc), "upstream_error")
            return
        except Exception as exc:  # noqa: BLE001
            self._send_error(500, f"Bridge failure: {exc}")
            return

        self._send_json(
            200,
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )

    def _translate(self, model: str, system_prompt: Optional[str], user_text: str) -> str:
        """
        Runs one translation through a fresh opencode session.

        A fresh session per call keeps requests independent and stops the
        model's conversational history from contaminating later translations.
        """
        session_id = self.opencode.create_session(title=f"bridge-{model}")
        try:
            return self.opencode.send_message(
                session_id=session_id,
                text=user_text,
                model=model,
                system=system_prompt,
            )
        finally:
            self.opencode.delete_session(session_id)


def build_server(
    port: int,
    opencode_url: str,
    default_model: str,
    password: Optional[str] = None,
) -> ThreadingHTTPServer:
    handler = type(
        "BoundBridgeHandler",
        (BridgeHandler,),
        {
            "opencode": OpenCodeServerClient(opencode_url, password=password),
            "default_model": strip_provider_prefix(default_model),
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    return server


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible bridge in front of the opencode CLI server."
    )
    parser.add_argument("--port", type=int, default=4096, help="Port for the bridge (default: 4096)")
    parser.add_argument(
        "--opencode-url",
        default=DEFAULT_OPENCODE_SERVER,
        help=f"URL of the running `opencode serve` (default: {DEFAULT_OPENCODE_SERVER})",
    )
    parser.add_argument(
        "--model",
        default="mimo-v2.6-flash-free",
        help="Default opencode model ID when the request omits one",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="OPENCODE_SERVER_PASSWORD, if you protected the opencode server with basic auth",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the opencode server is reachable, then exit",
    )
    args = parser.parse_args(argv)

    client = OpenCodeServerClient(args.opencode_url, password=args.password)

    if args.check:
        try:
            info = client.health()
            print(f"opencode server reachable at {args.opencode_url}: {info}")
            return 0
        except OpenCodeClientError as exc:
            print(f"Cannot reach opencode server: {exc}", file=sys.stderr)
            return 1

    # Fail fast with a clear message if the official client is not running.
    try:
        info = client.health()
        print(f"[bridge] connected to opencode server: {info}")
    except OpenCodeClientError as exc:
        print(f"[bridge] WARNING: {exc}", file=sys.stderr)
        print(
            "[bridge] Start it first in another terminal:\n"
            f"[bridge]     opencode serve --port 4096\n",
            file=sys.stderr,
        )

    server = build_server(
        port=args.port,
        opencode_url=args.opencode_url,
        default_model=args.model,
        password=args.password,
    )
    print(f"[bridge] OpenAI-compatible endpoint: http://127.0.0.1:{args.port}/v1")
    print(f"[bridge] default model: {strip_provider_prefix(args.model)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[bridge] shutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
