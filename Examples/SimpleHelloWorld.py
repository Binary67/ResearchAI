from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Codex import CodexClient


def main() -> None:
    client = CodexClient()

    try:
        session = client.start_session(cwd=str(PROJECT_ROOT), dangerous=True)
        result = session.prompt(
            "Create a file named HelloWorld.py in the project root. "
            "It should print 'Hello, World!' when run."
        )
        print(result.final_text)
    finally:
        client.close()


if __name__ == "__main__":
    main()
