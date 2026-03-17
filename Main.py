from Optimizer.Orchestrator import run_experiment

run_experiment(
    dataset_slug="yasserh/housing-prices-dataset",
    data_file="Housing.csv",
    target="price",
    max_iterations=5,
    score_threshold=0.0,
)
