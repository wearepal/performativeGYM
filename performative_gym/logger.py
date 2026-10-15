import json
import os
import time
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import numpy as np
import wandb

if TYPE_CHECKING:
    from mlflow.entities import Metric

__all__ = ["Logger", "Log"]


class Log(Enum):
    OFFLINE = auto()
    WANDB = auto()
    ML_FLOW = auto()


class Logger:
    def __init__(
        self,
        project: str,
        group: str,
        config: dict[str, Any],
        name: str,
        log_type: Log = Log.WANDB,
        login: bool = True,
    ):
        self.log_type = log_type
        self.login = login

        match log_type:
            case Log.WANDB:
                wandb.init(project=project, group=group, name=name, config=config)
            case Log.OFFLINE:
                self.save_dir = os.path.join("data", group, name + ".json")
                self.data = {}
                if not os.path.exists(os.path.join("data", group)):
                    os.mkdir(os.path.join("data", group))
            case Log.ML_FLOW:
                import mlflow
                if self.login:
                    mlflow.login()
                    mlflow.set_experiment(project)
                    mlflow.start_run(run_name=name)
                mlflow.set_tag("group", group)
                mlflow.log_params(config)
                # `mlflow.log_metrics` makes two blocking `get_run` requests on every
                # call, which against a remote server costs more than a training step.
                # Metrics are instead buffered and sent with `MlflowClient.log_batch`,
                # which makes a single request per batch.
                run = mlflow.active_run()  # `set_tag` started one if none was active
                assert run is not None
                self._mlflow_client = mlflow.MlflowClient()
                self._mlflow_run_id = run.info.run_id
                self._mlflow_metrics: list[Metric] = []

    def finish(self) -> None:
        match self.log_type:
            case Log.WANDB:
                wandb.finish()
            case Log.OFFLINE:
                with open(self.save_dir, "w") as json_file:
                    # Dump the dictionary into the file in JSON format
                    json.dump(self.data, json_file, indent=4)
                # pd.DataFrame(self.data).to_csv(self.save_dir, index=False)
            case Log.ML_FLOW:
                import mlflow
                self._flush_mlflow_metrics()
                if self.login:
                    mlflow.end_run()

    def log(self, data: dict[str, Any], step: int) -> None:
        match self.log_type:
            case Log.WANDB:
                wandb.log(data)
            case Log.OFFLINE:
                for key, value in data.items():
                    if key in self.data.keys():
                        self.data[key].append(value)
                    else:
                        self.data[key] = [value]
            case Log.ML_FLOW:
                from mlflow.entities import Metric
                from mlflow.utils.validation import MAX_METRICS_PER_BATCH

                timestamp = int(time.time() * 1000)
                self._mlflow_metrics.extend(
                    Metric(key, float(value), timestamp, step)
                    for key, value in data.items()
                )
                if len(self._mlflow_metrics) >= MAX_METRICS_PER_BATCH:
                    self._flush_mlflow_metrics()

    def _flush_mlflow_metrics(self) -> None:
        from mlflow.utils.validation import MAX_METRICS_PER_BATCH

        metrics = self._mlflow_metrics
        for start in range(0, len(metrics), MAX_METRICS_PER_BATCH):
            self._mlflow_client.log_batch(
                self._mlflow_run_id,
                metrics=metrics[start : start + MAX_METRICS_PER_BATCH],
            )
        self._mlflow_metrics = []

    def log_table(self, name: str, value: Any) -> None:
        match self.log_type:
            case Log.WANDB:
                wandb.log({name: wandb.Table(data=value)})
            case Log.OFFLINE:
                value = np.array(value).tolist()
                key = name
                if key in self.data.keys():
                    self.data[key].append(value)
                else:
                    self.data[key] = [value]
            case Log.ML_FLOW:
                import mlflow
                mlflow.log_table({name: value}, artifact_file=name + ".json")
