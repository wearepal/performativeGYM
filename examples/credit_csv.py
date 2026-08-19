"""Export the credit dataset to CSV, before and after the first performative shift.

Writes two files that line up row for row:

``<output_file>_raw.csv``
    The *Give Me Some Credit* dataset with only the structural preprocessing
    applied: NAs dropped, split into features and outcome, and class balanced.
    Not standardized.

``<output_file>_shifted.csv``
    The same rows after one step of PerfGD (reparametrization variant) followed
    by the strategic classification distribution shift.

Both are written in the original feature units: the pipeline only keeps the data
in standardized form, so both files are produced by inverting that with
:meth:`CreditDataset.unstandardize`.

Run with::

    python examples/credit_csv.py
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import numpy as np
import pandas as pd
import tyro
from jax import Array
from numpy import typing as npt

from examples.credit import CreditExp
from performative_gym import PerfGDReparam
from performative_gym.distribution_maps.datasets import CreditDataset
from performative_gym.optimizers import BaseOptimizer

jax.config.update("jax_enable_x64", True)


@dataclass
class CreditCsv:
    """Export the credit dataset before and after the first distribution shift."""

    out_dir: Path = Path("data")
    """Directory the CSV files are written to."""
    output_file: str = "credit"
    """Prefix of the written files: `<output_file>_raw.csv` and `<output_file>_shifted.csv`."""
    epsilon: float = 10
    """Epsilon for the data distribution shift; higher values lead to more significant shifts."""
    n: int = 120_000
    """Number of samples to use; the balanced credit dataset is smaller than this, so all of it."""
    seed: int = 10
    lr: float = 0.1
    """Learning rate for the optimizer."""
    model: Literal["NN", "logistic_regression"] = "NN"
    """Model type, either 'NN' for a neural network or 'logistic_regression'."""
    base_optimizer: BaseOptimizer = "GD"
    momentum: float = 0

    def run(self) -> None:
        # Reuse the experiment definition from `credit.py` for the model, the
        # loss and the distribution map, so this stays in sync with it.
        exp = CreditExp(
            epsilon=self.epsilon,
            n=self.n,
            seed=self.seed,
            lr=self.lr,
            model=self.model,
            base_optimizer=self.base_optimizer,
            momentum=self.momentum,
        )
        # `init_model` also sets `exp.h`, which `distribution_map` needs, so it
        # has to come first.
        params = exp.init_model()
        distribution_map = exp.distribution_map
        dataset = distribution_map.dataset

        optimizer = PerfGDReparam(
            params,
            lr=exp.lr,
            loss_fn=exp.loss_fn,
            proj_fn=exp.proj_fn,
            distr_map=distribution_map.sample,
            base_optimizer=exp.base_optimizer,
            momentum=exp.momentum,
        )

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._write_csv(
            dataset,
            distribution_map.x_0,
            distribution_map.y_0,
            self.out_dir / f"{self.output_file}_raw.csv",
        )

        # One PerfGD step, then the distribution shift it induces. Note that
        # `PerfGDReparam.step` ignores the batch it is handed and re-samples
        # inside the performative risk it differentiates, so there is no warm-up
        # phase: this first step is already a full performative gradient step.
        # The sample below is only kept to mirror the loop in `credit.py`.
        x_0, y_0 = distribution_map.sample(params)
        params = optimizer.step(params, x=x_0, y=y_0)
        x_1, y_1 = distribution_map.sample(params)

        self._write_csv(
            dataset, x_1, y_1, self.out_dir / f"{self.output_file}_shifted.csv"
        )

    def _write_csv(
        self,
        dataset: CreditDataset,
        features: Array | npt.NDArray,
        labels: Array | npt.NDArray,
        path: Path,
    ) -> None:
        """Undo the standardization and write the features and labels to `path`."""
        frame = pd.DataFrame(
            dataset.unstandardize(np.asarray(features)), columns=dataset.feature_names
        )
        frame["SeriousDlqin2yrs"] = np.asarray(labels)
        frame.to_csv(path, index=False)
        print(f"wrote {path}: {len(frame)} rows, {len(frame.columns)} columns")


if __name__ == "__main__":
    tyro.cli(CreditCsv, use_underscores=True).run()
