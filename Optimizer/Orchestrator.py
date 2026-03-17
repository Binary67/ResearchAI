import pickle
from io import StringIO
from pathlib import Path

from Agents import CodexClient
from Optimizer.Tracker import append_result
from Problem.Harness.DataPipeline import KaggleDataLoader, split_data
from Problem.Harness.EvaluationPipeline import ModelEvaluator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = PROJECT_ROOT / "Problem" / "Engine"
DATA_DIR = PROJECT_ROOT / "Problem" / "Data"
EVAL_SOURCE = PROJECT_ROOT / "Problem" / "Harness" / "EvaluationPipeline.py"
LOG_PATH = PROJECT_ROOT / "Optimizer" / "experiment_log.jsonl"


def _build_data_card(df):
    buf = StringIO()
    df.info(buf=buf)
    return (
        f"Shape: {df.shape}\n\n"
        f"Dtypes:\n{buf.getvalue()}\n\n"
        f"Head:\n{df.head().to_string()}\n\n"
        f"Describe:\n{df.describe().to_string()}"
    )


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
    data_card = _build_data_card(train_df)
    eval_source = EVAL_SOURCE.read_text()

    client = CodexClient()
    try:
        for iteration in range(1, max_iterations + 1):
            session = client.start_session(cwd=str(ENGINE_DIR), dangerous=True)

            # Turn 1 — EDA & Propose
            session.prompt(
                f"Here is a data card for the training data:\n\n"
                f"{data_card}\n\n"
                f"The evaluation code that will score your model:\n\n"
                f"```python\n{eval_source}\n```\n\n"
                f"The target column is `{target}`.\n"
                f"Explore the training data at ../Data/train.csv, do EDA, "
                f"and propose your modeling approach."
            )

            # Turn 2 — Implement
            session.prompt(
                "Implement your proposed plan. "
                "Save your trained model to model.pkl using pickle."
            )

            # Evaluate (orchestrator, not Codex)
            print(f"\n{'='*60}")
            print(f"Iteration {iteration} — Evaluation")
            print(f"{'='*60}")
            model_path = ENGINE_DIR / "model.pkl"
            score = None
            error = None
            try:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                score = ModelEvaluator(model, val_df, target).evaluate()
                print(f"MSE = {score}")
            except Exception as exc:
                error = str(exc)
                print(f"FAILED: {error}")

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
            )

            if score is not None and score <= score_threshold:
                break
    finally:
        client.close()
