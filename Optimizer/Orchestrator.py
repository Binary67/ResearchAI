import pickle
import random
import shutil
import sys
from pathlib import Path

from Agents import CodexClient
from Optimizer.Tracker import append_result
from Problem.Harness.DataPipeline import KaggleDataLoader, split_data
from Problem.Harness.EvaluationPipeline import ModelEvaluator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = PROJECT_ROOT / "Problem" / "Engine"
DATA_DIR = PROJECT_ROOT / "Problem" / "Data"
BEST_DIR = PROJECT_ROOT / "Problem" / "Best"
EVAL_SOURCE = PROJECT_ROOT / "Problem" / "Harness" / "EvaluationPipeline.py"
LOG_PATH = PROJECT_ROOT / "Optimizer" / "experiment_log.jsonl"


def _clean_engine():
    for item in ENGINE_DIR.iterdir():
        if item.name == "__init__.py":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def _read_code_files(directory):
    parts = []
    for p in sorted(Path(directory).glob("*.py")):
        parts.append(f"# {p.name}\n{p.read_text()}")
    return "\n\n".join(parts)


def run_experiment(
    dataset_slug,
    data_file,
    target,
    max_iterations=5,
    score_threshold=0.0,
    test_size=0.2,
    random_state=42,
):
    # 1. Load & split data
    loader = KaggleDataLoader()
    frames = loader.download(dataset_slug)
    df = frames[data_file]
    train_df, val_df = split_data(df, target, test_size=test_size, random_state=random_state)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "train.csv"
    train_df.to_csv(train_path, index=False)

    # 2. Prepare prompt context
    eval_source = EVAL_SOURCE.read_text()

    best_score = None
    client = CodexClient()
    try:
        for iteration in range(1, max_iterations + 1):
            # Decide explore vs exploit
            if max_iterations == 1 or best_score is None:
                explore_prob = 1.0
            else:
                explore_prob = 1 - (iteration - 1) / (max_iterations - 1)
            mode = "explore" if random.random() < explore_prob else "exploit"

            _clean_engine()

            if mode == "exploit":
                for item in BEST_DIR.iterdir():
                    if item.name == ".gitkeep":
                        continue
                    dest = ENGINE_DIR / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)

            session = client.start_session(cwd=str(ENGINE_DIR), dangerous=True)

            if mode == "explore":
                session.prompt(
                    f"The evaluation code that will score your model:\n\n"
                    f"```python\n{eval_source}\n```\n\n"
                    f"The target column is `{target}`.\n"
                    f"Explore the training data at ../Data/train.csv, do EDA, "
                    f"and propose your modeling approach."
                )
            else:
                code_listing = _read_code_files(ENGINE_DIR)
                session.prompt(
                    f"The evaluation code that will score your model:\n\n"
                    f"```python\n{eval_source}\n```\n\n"
                    f"The target column is `{target}`.\n\n"
                    f"You have a baseline solution that scored MSE = {best_score}. "
                    f"Its code is already in your working directory. Here it is:\n\n"
                    f"```\n{code_listing}\n```\n\n"
                    f"Analyze what this solution does, identify weaknesses, "
                    f"and propose improvements."
                )

            # Turn 2 — Implement
            if mode == "explore":
                impl_result = session.prompt(
                    "Implement your proposed plan. "
                    "Save your trained model to model.pkl using pickle."
                )
            else:
                impl_result = session.prompt(
                    "Implement your improvements. "
                    "Save your trained model to model.pkl using pickle."
                )

            if impl_result.files_changed == 0:
                session.prompt(
                    "You did not write any files. You MUST create code files and "
                    "save a trained model to model.pkl using pickle. "
                    "Write all necessary code and train the model now."
                )

            # Evaluate
            print(f"\n{'='*60}")
            print(f"Iteration {iteration} — {mode.upper()} — Evaluation")
            if best_score is not None:
                print(f"Best so far: MSE = {best_score}")
            print(f"{'='*60}")
            model_path = ENGINE_DIR / "model.pkl"
            score = None
            error = None
            problem_dir = str(ENGINE_DIR.parent)
            if problem_dir not in sys.path:
                sys.path.insert(0, problem_dir)
            try:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                score = ModelEvaluator(model, val_df, target).evaluate()
                print(f"MSE = {score}")
            except Exception as exc:
                error = str(exc)
                print(f"FAILED: {error}")

            # Update best
            if score is not None and (best_score is None or score < best_score):
                best_score = score
                if BEST_DIR.exists():
                    shutil.rmtree(BEST_DIR)
                BEST_DIR.mkdir(parents=True)
                for item in ENGINE_DIR.iterdir():
                    if item.name == "__init__.py":
                        continue
                    dest = BEST_DIR / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)
                print(f"New best! MSE = {best_score}")

            # Turn 3 — Summarize
            if score is not None:
                summary_result = session.prompt(
                    f"Your model scored MSE = {score}. "
                    f"Summarize what you did, what worked, and what failed."
                )
            else:
                summary_result = session.prompt(
                    f"Evaluation failed: {error}. "
                    f"Summarize what you attempted and what went wrong."
                )

            print(f"\nSummary: {summary_result.final_text}\n")

            append_result(
                LOG_PATH,
                iteration,
                score,
                summary_result.final_text,
                mode,
            )

            if score is not None and score <= score_threshold:
                break
    finally:
        client.close()
