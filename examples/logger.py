from typing import Any
import os
import wandb
import json

class Logger:
    def __init__(self, project: str, group: str, config: dict[str, Any], name: str = None, upload: bool = True):
        self.upload = upload

        if self.upload:
            wandb.init(project=project, group=group, name=name, config=config)
            self.config = wandb.config
        else:
            self.save_dir = os.path.join('data', group, name + '.json')
            self.config = config
            self.data = {}
            if not os.path.exists(os.path.join('data', group)):
                os.mkdir(os.path.join('data', group))

    def update_config(self, config: dict[str, Any]) -> None:
        if self.upload:
            wandb.config.update(config)
            self.config = wandb.config
        else:
            self.config = self.config.update(config)

    def finish(self) -> None:
        if self.upload:
            wandb.finish()
        else:
            with open(self.save_dir, 'w') as json_file:
                # Dump the dictionary into the file in JSON format
                json.dump(self.data, json_file, indent=4)
            #pd.DataFrame(self.data).to_csv(self.save_dir, index=False)

    def log(self, data: dict[str, Any]) -> None:
        if self.upload:
            wandb.log(data)
        else:
            for key, value in data.items():
                if key in self.data.keys():
                    self.data[key].append(value)
                else:
                    self.data[key] = [value]
