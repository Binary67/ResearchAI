from sklearn.metrics import mean_squared_error


class ModelEvaluator:
    def __init__(self, model, validation_data, y_label: str):
        self.model = model
        self.validation_data = validation_data
        self.y_label = y_label

    def evaluate(self) -> float:
        X = self.validation_data.drop(columns=[self.y_label])
        y_true = self.validation_data[self.y_label]
        y_pred = self.model.predict(X)
        return float(mean_squared_error(y_true, y_pred))
