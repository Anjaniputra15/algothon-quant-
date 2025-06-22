import mlflow
from mlflow.tracking import MlflowClient

BEST_METRIC = "mean_return"  # Change to your main metric
EXPERIMENT_NAME = "Default"
MODEL_NAME = "algothon-quant-model"

client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found.")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"attributes.status = 'FINISHED' and metrics.{BEST_METRIC} IS NOT NULL",
    order_by=[f"metrics.{BEST_METRIC} DESC"],
    max_results=1,
)

if not runs:
    raise RuntimeError("No completed runs with the target metric found.")

best_run = runs[0]
best_run_id = best_run.info.run_id
print(f"Best run: {best_run_id}, {BEST_METRIC}={best_run.data.metrics.get(BEST_METRIC)}")

# Register the model if not already
model_uri = f"runs:/{best_run_id}/model"
result = mlflow.register_model(model_uri, MODEL_NAME)

# Promote to Production
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=result.version,
    stage="Production",
    archive_existing_versions=True,
)
print(f"Promoted model version {result.version} to Production.") 