import pandas as pd
from sklearn.linear_model import LinearRegression
from Problem.Harness.EvaluationPipeline import ModelEvaluator

# Create sample training data
train = pd.DataFrame({"x1": [1, 2, 3, 4, 5], "x2": [2, 4, 6, 8, 10], "y": [3, 6, 9, 12, 15]})

# Train a simple model
model = LinearRegression()
model.fit(train[["x1", "x2"]], train["y"])

# Create validation data
validation = pd.DataFrame({"x1": [6, 7, 8], "x2": [12, 14, 16], "y": [18, 21, 24]})

# Evaluate
evaluator = ModelEvaluator(model, validation, y_label="y")
mse = evaluator.evaluate()
print(f"MSE: {mse}")
