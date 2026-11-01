import sys
import mlflow
from mlflow import MlflowClient
from config import MLFLOW_URI, REGISTERED_MODEL, LAST_RUN_FILE
from registry import register_run, set_champion

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

new_run_id = LAST_RUN_FILE.read_text().strip()
new = client.get_run(new_run_id).data.metrics
print("Nouveau modèle :", {k: round(v, 3) for k, v in new.items() if k.startswith("val_")})

try:
    champ_version = client.get_model_version_by_alias(REGISTERED_MODEL, "champion")
    champ = client.get_run(champ_version.run_id).data.metrics
    print("Champion actuel :", {k: round(v, 3) for k, v in champ.items() if k.startswith("val_")})
    better = (new["val_recall"] > champ["val_recall"]) and (new["val_map50"] >= champ["val_map50"] - 0.03)
except Exception:
    print("Pas de champion existant : le nouveau modèle devient champion.")
    better = True

if better:
    version = register_run(new_run_id)
    set_champion(version)
    print("NOUVEAU CHAMPION")
else:
    print("Champion conservé (le nouveau modèle n'est pas meilleur).")
sys.exit(0)