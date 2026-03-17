import json
from pathlib import Path


def append_result(path, iteration, score, summary):
    record = {"iteration": iteration, "score": score, "summary": summary}
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_results(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
