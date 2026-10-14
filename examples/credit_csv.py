"""Export the credit dataset to CSV, before and after two performative shifts.

Writes three files that line up row for row:

``<output_file>_raw.csv``
    The *Give Me Some Credit* dataset with only the structural preprocessing
    applied: NAs dropped, split into features and outcome, and class balanced.
    Not standardized.

``<output_file>_shifted1.csv``
    The same rows after the strategic classification distribution shift in
    response to a model trained on the raw data.

``<output_file>_shifted2.csv``
    The same rows after the shift in response to the model trained a second
    time, starting from the first model.

How the model is trained before each shift depends on ``--optimizer``:

``RRM``
    Repeated risk minimization: the model is retrained on the data it currently
    faces, without anticipating the shift, i.e. first on the raw data and then
    on the data of the first shift. This gives the successive shifts that a
    naively retrained model is exposed to.

``PerfGDReparam``
    Performative gradient descent: the model is trained on the data shifted in
    response to itself, anticipating the shift. The second training continues
    from the first model, so once training has converged, the two shifts are
    nearly the same.

All files are written in the original feature units: the pipeline only keeps
the data in standardized form, so the files are produced by inverting that with
:meth:`CreditDataset.unstandardize`.

Run with::

    python examples/credit_csv.py --optimizer RRM
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
from tqdm.auto import trange

from examples.credit import CreditExp
from performative_gym import RRM, Optimizer, PerfGDReparam
from performative_gym.distribution_maps.datasets import CreditDataset
from performative_gym.optimizers import BaseOptimizer

jax.config.update("jax_enable_x64", True)


@dataclass
class CreditCsv:
    """Export the credit dataset before and after two distribution shifts."""

    out_dir: Path = Path("data")
    """Directory the CSV files are written to."""
    epsilon: float = 10
    """Epsilon for the data distribution shift; higher values lead to more significant shifts."""
    n: int = 120_000
    """Number of samples to use; the balanced credit dataset is smaller than this, so all of it."""
    seed: int = 10
    optimizer: Literal["RRM", "PerfGDReparam"] = "PerfGDReparam"
    """How the model is trained before each shift; see the module docstring."""
    iterations: int = 5000
    """Gradient steps in each of the two trainings. For RRM, these are the steps
    of the one retraining on the current data."""
    base_optimizer: BaseOptimizer = "GD"
    """Base optimizer of PerfGDReparam; RRM always uses plain gradient descent."""
    momentum: float = 0
    """Momentum of PerfGDReparam's base optimizer; unused by RRM."""
    output_file: str = "credit"
    """Prefix of the written files: `<output_file>_raw.csv`, `<output_file>_shifted1.csv`
    and `<output_file>_shifted2.csv`."""
    model: Literal["NN", "logistic_regression"] = "NN"
    """Model type, either 'NN' for a neural network or 'logistic_regression'."""
    lr: float = 0.1
    """Learning rate for the optimizer."""

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

        optimizer: Optimizer
        match self.optimizer:
            case "PerfGDReparam":
                optimizer = PerfGDReparam(
                    params,
                    lr=exp.lr,
                    loss_fn=exp.loss_fn,
                    proj_fn=exp.proj_fn,
                    distr_map=distribution_map.sample,
                    base_optimizer=exp.base_optimizer,
                    momentum=exp.momentum,
                )
                steps = self.iterations
            case "RRM":
                # A single RRM step retrains on the batch it is handed, running
                # all the gradient steps of one training.
                optimizer = RRM(
                    params,
                    lr=exp.lr,
                    loss_fn=exp.loss_fn,
                    proj_fn=exp.proj_fn,
                    tol=None,
                    iterations=self.iterations,
                )
                steps = 1

        self.out_dir.mkdir(parents=True, exist_ok=True)
        x, y = distribution_map.x_0, distribution_map.y_0
        self._write_csv(dataset, x, y, self.out_dir / f"{self.output_file}_raw.csv")

        # Before anything is deployed, the model faces the raw data. Each round
        # trains on the data the model currently faces, then deploys it, which
        # shifts the data in response. `PerfGDReparam.step` ignores the batch it
        # is handed and re-samples inside the performative risk it
        # differentiates, so for it the batch only matters for the accuracies.
        for shift in (1, 2):
            for _ in trange(steps, desc=f"training before shift {shift}"):
                params = optimizer.step(params, x=x, y=y)
            accuracy_before = exp.accuracy(params, x, y).item()
            x, y = distribution_map.sample(params)
            accuracy_after = exp.accuracy(params, x, y).item()
            print(
                f"shift {shift}: accuracy {accuracy_before:.4f} before the shift, "
                f"{accuracy_after:.4f} after the shift"
            )
            self._write_csv(
                dataset, x, y, self.out_dir / f"{self.output_file}_shifted{shift}.csv"
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
