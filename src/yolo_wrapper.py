import mlflow.pyfunc


class YoloWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from ultralytics import YOLO
        self.model = YOLO(context.artifacts["weights"])

    def predict(self, context, model_input, params=None):
        conf = (params or {}).get("conf", 0.25)
        return self.model.predict(model_input, conf=conf, verbose=False)