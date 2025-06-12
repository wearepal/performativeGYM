from enum import Enum, auto
from typing import Any
import os
import wandb
import mlflow
import json

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
    ):
        self.log_type = log_type

        match log_type:
            case Log.WANDB:
                wandb.init(project=project, group=group, name=name, config=config)
                self.config = wandb.config
            case Log.OFFLINE:
                self.save_dir = os.path.join("data", group, name + ".json")
                self.config = config
                self.data = {}
                if not os.path.exists(os.path.join("data", group)):
                    os.mkdir(os.path.join("data", group))
            case Log.ML_FLOW:
                mlflow.set_experiment(project)
                mlflow.start_run(run_name=name)
                mlflow.log_params(config)

    def update_config(self, config: dict[str, Any]) -> None:
        match self.log_type:
            case Log.WANDB:
                wandb.config.update(config)
                self.config = wandb.config
            case Log.OFFLINE:
                self.config.update(config)
            case Log.ML_FLOW:
                mlflow.log_params(config)
                self.config.update(config)

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
                mlflow.end_run()

    def log(self, data: dict[str, Any]) -> None:
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
                mlflow.log_metrics(data)
