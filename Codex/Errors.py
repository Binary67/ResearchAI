class CodexError(Exception):
    """Base exception for the Codex wrapper."""


class CodexProcessError(CodexError):
    """Raised when the app-server process cannot be started or exits unexpectedly."""


class CodexRpcError(CodexError):
    """Raised when app-server returns a JSON-RPC error response."""

    def __init__(self, code: int, message: str):
        super().__init__(f"Codex RPC error {code}: {message}")
        self.code = code
        self.message = message


class CodexTurnError(CodexError):
    """Raised when a turn finishes unsuccessfully."""

    def __init__(self, status: str, message: str, thread_id: str, turn_id: str):
        super().__init__(f"Codex turn {status}: {message}")
        self.status = status
        self.message = message
        self.thread_id = thread_id
        self.turn_id = turn_id
