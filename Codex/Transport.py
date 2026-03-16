from __future__ import annotations

import json
import queue
import subprocess
import threading
from collections import deque
from itertools import count
from typing import Any

from Codex.Errors import CodexProcessError, CodexRpcError, CodexTurnError
from Codex.Types import PromptResult


_CLIENT_INFO = {
    "name": "researchai",
    "title": "ResearchAI",
    "version": "0.1.0",
}


class AppServerTransport:
    def __init__(self, executable: str = "codex"):
        self._executable = executable
        self._process: subprocess.Popen[str] | None = None
        self._stderr_lines: deque[str] = deque(maxlen=50)
        self._request_ids = count(1)
        self._pending_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending: dict[int, queue.Queue[dict[str, Any]]] = {}
        self._notifications: queue.Queue[dict[str, Any]] = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._initialized = False

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            if not self._initialized:
                self.request("initialize", {"clientInfo": _CLIENT_INFO})
                self.notify("initialized", {})
                self._initialized = True
            return

        self._notifications = queue.Queue()
        self._pending = {}
        self._stderr_lines.clear()

        try:
            self._process = subprocess.Popen(
                [self._executable, "app-server"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as exc:
            raise CodexProcessError(f"Failed to start `{self._executable} app-server`: {exc}") from exc

        if self._process.stdin is None or self._process.stdout is None or self._process.stderr is None:
            self.close()
            raise CodexProcessError("Failed to create stdio pipes for `codex app-server`.")

        self._initialized = False
        self._reader_thread = threading.Thread(target=self._read_stdout, name="codex-stdout", daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, name="codex-stderr", daemon=True)
        self._reader_thread.start()
        self._stderr_thread.start()

        self.request("initialize", {"clientInfo": _CLIENT_INFO})
        self.notify("initialized", {})
        self._initialized = True

    def close(self) -> None:
        process = self._process
        self._process = None
        self._initialized = False

        if process is None:
            return

        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()

        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._process is None or self._process.poll() is not None:
            self.start()
        elif not self._initialized and method != "initialize":
            self.start()

        request_id = next(self._request_ids)
        response_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[request_id] = response_queue

        self._write_message({"method": method, "id": request_id, "params": params or {}})

        while True:
            try:
                response = response_queue.get(timeout=0.1)
                break
            except queue.Empty:
                self._raise_if_process_exited()

        if "error" in response:
            error = response["error"]
            raise CodexRpcError(error.get("code", -1), error.get("message", "Unknown JSON-RPC error"))

        return response.get("result", {})

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        if self._process is None or self._process.poll() is not None:
            self.start()
        elif not self._initialized and method != "initialized":
            self.start()
        self._write_message({"method": method, "params": params or {}})

    def start_thread(self, cwd: str, dangerous: bool) -> str:
        params: dict[str, Any] = {"cwd": cwd}
        params.update(_dangerous_overrides(dangerous))
        result = self.request("thread/start", params)
        thread = result.get("thread") or {}
        thread_id = thread.get("id")
        if not isinstance(thread_id, str) or not thread_id:
            raise CodexProcessError("`thread/start` did not return a thread id.")
        return thread_id

    def resume_thread(self, thread_id: str, cwd: str | None, dangerous: bool) -> str:
        params: dict[str, Any] = {"threadId": thread_id}
        if cwd is not None:
            params["cwd"] = cwd
        params.update(_dangerous_overrides(dangerous))
        result = self.request("thread/resume", params)
        thread = result.get("thread") or {}
        resumed_id = thread.get("id")
        if not isinstance(resumed_id, str) or not resumed_id:
            raise CodexProcessError("`thread/resume` did not return a thread id.")
        return resumed_id

    def prompt(self, thread_id: str, text: str, cwd: str | None, dangerous: bool) -> PromptResult:
        self._drain_notifications()

        params: dict[str, Any] = {
            "threadId": thread_id,
            "input": [{"type": "text", "text": text}],
        }
        if cwd is not None:
            params["cwd"] = cwd
        params.update(_dangerous_overrides(dangerous))

        result = self.request("turn/start", params)
        turn = result.get("turn") or {}
        turn_id = turn.get("id")
        if not isinstance(turn_id, str) or not turn_id:
            raise CodexProcessError("`turn/start` did not return a turn id.")

        return self._collect_turn_result(thread_id=thread_id, turn_id=turn_id)

    def _collect_turn_result(self, thread_id: str, turn_id: str) -> PromptResult:
        agent_messages: list[dict[str, Any]] = []
        delta_text_by_item: dict[str, str] = {}

        while True:
            try:
                notification = self._notifications.get(timeout=0.1)
            except queue.Empty:
                self._raise_if_process_exited()
                continue

            method = notification.get("method")
            params = notification.get("params") or {}

            if method == "item/completed":
                item = params.get("item") or {}
                if item.get("type") == "agentMessage":
                    agent_messages.append(item)
                continue

            if method == "item/agentMessage/delta":
                item_id = params.get("itemId") or params.get("id")
                delta_text = _extract_delta_text(params)
                if isinstance(item_id, str) and delta_text:
                    delta_text_by_item[item_id] = delta_text_by_item.get(item_id, "") + delta_text
                continue

            if method != "turn/completed":
                continue

            completed_turn = params.get("turn") or {}
            if completed_turn.get("id") != turn_id:
                continue

            status = completed_turn.get("status", "unknown")
            if status != "completed":
                error = completed_turn.get("error") or {}
                message = error.get("message", f"Turn finished with status `{status}`.")
                raise CodexTurnError(status=status, message=message, thread_id=thread_id, turn_id=turn_id)

            return PromptResult(
                thread_id=thread_id,
                turn_id=turn_id,
                status=status,
                final_text=_choose_final_text(agent_messages, delta_text_by_item),
            )

    def _drain_notifications(self) -> None:
        while True:
            try:
                self._notifications.get_nowait()
            except queue.Empty:
                return

    def _write_message(self, message: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise CodexProcessError("`codex app-server` is not running.")

        try:
            payload = json.dumps(message, separators=(",", ":"))
            with self._write_lock:
                process.stdin.write(f"{payload}\n")
                process.stdin.flush()
        except OSError as exc:
            raise CodexProcessError(f"Failed to write to `codex app-server`: {exc}") from exc

    def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return

        for line in process.stdout:
            message_text = line.strip()
            if not message_text:
                continue

            try:
                message = json.loads(message_text)
            except json.JSONDecodeError:
                continue

            if "id" in message:
                request_id = message["id"]
                with self._pending_lock:
                    response_queue = self._pending.pop(request_id, None)
                if response_queue is not None:
                    response_queue.put(message)
                continue

            self._notifications.put(message)

    def _read_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return

        for line in process.stderr:
            message_text = line.rstrip()
            if message_text:
                self._stderr_lines.append(message_text)

    def _raise_if_process_exited(self) -> None:
        process = self._process
        if process is None:
            raise CodexProcessError("`codex app-server` is not running.")

        return_code = process.poll()
        if return_code is None:
            return

        stderr_output = "\n".join(self._stderr_lines)
        detail = f" Exit code: {return_code}."
        if stderr_output:
            detail = f"{detail} stderr:\n{stderr_output}"
        raise CodexProcessError(f"`codex app-server` exited unexpectedly.{detail}")


def _dangerous_overrides(dangerous: bool) -> dict[str, Any]:
    if not dangerous:
        return {}
    return {
        "approvalPolicy": "never",
        "sandboxPolicy": {"type": "dangerFullAccess"},
    }


def _extract_delta_text(params: dict[str, Any]) -> str:
    for key in ("delta", "textDelta", "text"):
        value = params.get(key)
        if isinstance(value, str):
            return value
    return ""


def _choose_final_text(agent_messages: list[dict[str, Any]], delta_text_by_item: dict[str, str]) -> str:
    final_messages = [item for item in agent_messages if item.get("phase") == "final_answer"]
    if final_messages:
        text = final_messages[-1].get("text")
        if isinstance(text, str):
            return text

    if agent_messages:
        text = agent_messages[-1].get("text")
        if isinstance(text, str):
            return text

    if delta_text_by_item:
        return next(reversed(delta_text_by_item.values()))

    return ""
