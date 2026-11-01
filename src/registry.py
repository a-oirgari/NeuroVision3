import argparse
from pathlib import Path
import mlflow
from mlflow import MlflowClient
from config import MLFLOW_URI, REGISTERED_MODEL, ROOT
from yolo_wrapper import YoloWrapper

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()


def register_run(run_id: str):
    weights = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="weights/best.pt")
    with mlflow.start_run(run_id=run_id):
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=YoloWrapper(),
            artifacts={"weights": weights},
            code_paths=[str(Path(__file__).parent / "yolo_wrapper.py")],
            pip_requirements=["ultralytics", "mlflow"],
            registered_model_name=REGISTERED_MODEL,
        )
    return info.registered_model_version


def set_champion(version: str):
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", version)
    print(f"Alias 'champion' -> version {version}")


