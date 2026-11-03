import wandb
from bem.Logger import Logger


class WandbLogger(Logger):
    """Weights & Biases logger implementing the bem Logger interface."""

    def __init__(self, project=None, name=None, entity=None):
        super().__init__()
        self.project = project or 'htfm'
        self.name = name
        self.entity = entity
        self.run = None
        self._step = 0

    def initialize(self, p):
        self.run = wandb.init(
            project=self.project,
            name=self.name,
            entity=self.entity,
            config=p,
            reinit=True,
        )

    def set_values(self, value_dict):
        """Restore logged values from a checkpoint.

        value_dict is a nested dict like {folder: {metric: array_of_values}}.
        We log a summary so wandb knows the latest state.
        """
        if self.run is None:
            return
        for folder, metrics in value_dict.items():
            if not isinstance(metrics, dict):
                continue
            for k, v in metrics.items():
                # skip large arrays (losses_batch history etc.)
                try:
                    if hasattr(v, '__len__') and len(v) > 1:
                        # log only the last value as summary
                        wandb.run.summary[f"{folder}/{k}"] = float(v[-1])
                    else:
                        wandb.run.summary[f"{folder}/{k}"] = v
                except (TypeError, ValueError, IndexError):
                    pass

    def log(self, data_type, data):
        if self.run is None:
            return
        # scalar or string values
        try:
            val = float(data)
        except (TypeError, ValueError):
            val = data

        if data_type == 'current_batch':
            self._step = int(data)
            wandb.log({data_type: val}, step=self._step)
        elif data_type == 'current_epoch':
            wandb.log({data_type: val}, step=self._step)
        else:
            wandb.log({data_type: val}, step=self._step)

    def stop(self):
        if self.run is not None:
            wandb.finish()
            self.run = None
