"""Load and preprocess Kaggle credit dataset."""

import os

import jax
import numpy as np
from numpy import typing as npt
import pandas as pd
from sklearn.preprocessing import StandardScaler


class CreditDataset:
    """
    Credit dataset loader.

    This class provides a utility to load and preprocess the
    *Give Me Some Credit* dataset for use in performative prediction experiments,
    particularly in strategic classification settings.

    The dataset consists of feature vectors :math:`x \\in \\mathbb{R}^d` describing
    individual financial characteristics, and binary labels

    .. math::

        y \\in \\{0,1\\},

    where :math:`y=1` indicates default (``SeriousDlqin2yrs``).

    The class performs several preprocessing steps to obtain a clean and balanced
    dataset suitable for learning algorithms.

    Processing pipeline
    -------------------
    Given the raw dataset, the following transformations are applied:

    1. **Missing values removal**:
       All rows containing missing entries are dropped.

    2. **Class balancing**:
       The dataset is subsampled to obtain equal numbers of positive and negative
       examples:

       .. math::

           |\\{i : y_i = 1\\}| = |\\{i : y_i = 0\\}|.

    3. **Shuffling**:
       The dataset is randomly permuted using a JAX PRNG key.

    4. **Feature standardization**:
       Features are normalized to zero mean and unit variance, using statistics
       computed on the balanced subset:

       .. math::

           x \\leftarrow \\frac{x - \\mu}{\\sigma}.

    Steps 1 to 3 are available on their own via :meth:`load_raw_data`, and
    step 4 can be undone via :meth:`unstandardize`.

    Parameters
    ----------
    datafile : str, default="credit_data.zip"
        Name of the file containing the dataset. It is expected to be located in
        the same directory as this module.

    seed : int
        Random seed used for shuffling the dataset.

    Attributes
    ----------
    datapath : str
        Absolute path to the dataset file.

    seed : int
        Random seed used for data processing.

    features : ndarray of shape (n, d)
        Preprocessed feature matrix.

    labels : ndarray of shape (n,)
        Binary labels indicating default status.

    standard_scaler : sklearn.preprocessing.StandardScaler
        The scaler fitted on the features. Its ``mean_`` and ``scale_``
        attributes can be used to reproduce the standardization for new inputs
        at inference time.

    feature_names : list of str
        Names of the features, in the same column order the scaler expects.

    Properties
    ----------
    num_agents : int
        Number of samples in the dataset:

        .. math::

            n = \\text{features.shape}[0].

    num_features : int
        Number of features per sample:

        .. math::

            d = \\text{features.shape}[1].

    feature_mean : ndarray of shape (d,)
        Per-feature mean used for standardization.

    feature_std : ndarray of shape (d,)
        Per-feature standard deviation used for standardization.

    Methods
    -------
    load_data(seed: int) -> tuple[ndarray, ndarray]
        Load, preprocess, balance, and shuffle the dataset. Returns the feature
        matrix and label vector.

    load_raw_data(seed: int) -> tuple[DataFrame, ndarray]
        Steps 1 to 3 only, so the features are in their original units but
        already in the final row order.

    unstandardize(features: ndarray) -> ndarray
        Inverse of the standardization: rescale by ``feature_std`` and shift by
        ``feature_mean``.

    __len__() -> int
        Return the number of samples in the dataset.

    Notes
    -----
    - The balancing step reduces the dataset size to twice the number of positive
      examples, ensuring equal class proportions.
    - No constant column is appended to the features. An intercept is a model
      parameter, not a feature, and a performative distribution map such as
      :class:`~.StrategicClassification` shifts every feature column: a constant
      column would be shifted along with the real features, which is not a
      meaningful strategic response. Models needing an intercept must carry it
      in their parameters.
    - Shuffling is performed using JAX for consistency with the rest of the codebase.
    """

    def __init__(self, datafile: str = "credit_data.zip", *, seed: int):
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        datapath = os.path.join(cur_dir, datafile)

        self.datapath = datapath
        self.seed = seed
        self.standard_scaler = StandardScaler()
        self.features, self.labels, self.feature_names = self._load_data(self.seed)

    @property
    def num_agents(self):
        """Compute number of agents in the dataset."""
        return self.features.shape[0]

    @property
    def num_features(self):
        """Compute number of features for each agent."""
        return self.features.shape[1]

    @property
    def feature_mean(self) -> npt.NDArray:
        """Per-feature mean used for standardization (raw features, no bias)."""
        mean = self.standard_scaler.mean_
        assert mean is not None, "Scaler has not been fitted yet."
        return mean

    @property
    def feature_std(self) -> npt.NDArray:
        """Per-feature standard deviation used for standardization (raw features, no bias)."""
        scale = self.standard_scaler.scale_
        assert scale is not None, "Scaler has not been fitted yet."
        return scale

    def load_data(self, seed: int) -> tuple[npt.NDArray, npt.NDArray]:
        """Load the (features, labels) for the given seed.

        Kept for backwards compatibility; delegates to :meth:`_load_data` and
        drops the feature names from the return value.
        """
        features, labels, _ = self._load_data(seed)
        return features, labels

    def load_raw_data(self, seed: int) -> tuple[pd.DataFrame, npt.NDArray]:
        """Load the balanced and shuffled (features, labels), in the original units.

        Applies steps 1 to 3 of the pipeline only -- dropping NAs, splitting off
        the outcome, balancing the classes and shuffling -- so the rows line up
        with :attr:`features` but are not standardized.
        """
        key = jax.random.PRNGKey(seed)

        data = pd.read_csv(self.datapath, index_col=0)
        data.dropna(inplace=True)

        features_df = data.drop("SeriousDlqin2yrs", axis=1)
        outcomes = np.array(data["SeriousDlqin2yrs"])  # 120000 samples

        # balance classes
        default_indices = np.where((outcomes == 1))[0]  # 8000
        other_indices = np.where((outcomes == 0))[0][: len(default_indices)]  # 112000
        indices = np.concatenate((default_indices, other_indices))

        # shuffle arrays
        shuffled = np.asarray(jax.random.permutation(key, indices))

        # `shuffled` holds positional indices, so index positionally: the
        # DataFrame index has gaps in it after dropping the NAs.
        return features_df.iloc[shuffled], outcomes[shuffled]

    def _load_data(self, seed: int) -> tuple[npt.NDArray, npt.NDArray, list[str]]:
        features_df, outcomes_balanced = self.load_raw_data(seed)
        feature_names = list(features_df.columns)

        # zero mean, unit variance (fitted on the balanced subset). The row
        # slice in `load_raw_data` makes `to_numpy()` come out Fortran-ordered;
        # `ascontiguousarray` keeps the scaler's summation order -- and hence
        # its mean and scale down to the last bit -- what it was before.
        features_balanced = self.standard_scaler.fit_transform(
            np.ascontiguousarray(features_df.to_numpy())
        )

        return features_balanced, outcomes_balanced, feature_names

    def __len__(self):
        return len(self.labels)

    def unstandardize(self, features: npt.NDArray) -> npt.NDArray:
        """Map standardized features back to the original units.

        Applies the inverse of the fitted scaler, i.e.
        ``x * feature_std + feature_mean``. This is the inverse of step 4 of the
        pipeline, up to floating point round-off; see
        :func:`test_standardization_roundtrip`.
        """
        return self.standard_scaler.inverse_transform(features)


def test_standardization_roundtrip(seed: int = 0, tolerance: float = 1e-10) -> None:
    """Check that :meth:`CreditDataset.unstandardize` recovers the original data.

    The standardized features are the only form in which the pipeline keeps the
    data around, so anything that wants the data in its original units (an
    export to CSV, say, or a shifted sample coming out of a distribution map)
    has to go back through the scaler. This checks that the round-trip loses
    nothing beyond floating point round-off.
    """
    dataset = CreditDataset(seed=seed)
    raw, raw_labels = dataset.load_raw_data(seed)
    assert list(raw.columns) == dataset.feature_names
    original = raw.to_numpy()

    # the standardized and the raw path have to agree on the row order
    assert np.array_equal(raw_labels, dataset.labels)

    reconstructed = dataset.unstandardize(dataset.features)
    assert reconstructed.shape == original.shape

    # Relative error, with a floor of 1 in the denominator: many columns are
    # small counts (and often exactly 0), where an absolute comparison is the
    # meaningful one.
    error = np.abs(reconstructed - original) / np.maximum(np.abs(original), 1.0)
    worst = error.max()
    assert worst < tolerance, (
        f"worst relative reconstruction error {worst:.3e} exceeds {tolerance:.0e} "
        f"(feature {dataset.feature_names[int(error.max(axis=0).argmax())]})"
    )


if __name__ == "__main__":
    seed = 0
    dataset = CreditDataset(seed=seed)
    original = dataset.load_raw_data(seed)[0].to_numpy()
    reconstructed = dataset.unstandardize(dataset.features)

    abs_err = np.abs(reconstructed - original)
    rel_err = abs_err / np.maximum(np.abs(original), 1.0)

    name_width = max(len(name) for name in dataset.feature_names)
    print(f"reconstruction error over {len(dataset)} rows (seed {seed}):\n")
    print(f"{'feature':{name_width}s} {'scale':>12s} {'max abs':>12s} {'max rel':>12s}")
    for i, name in enumerate(dataset.feature_names):
        print(
            f"{name:{name_width}s} {dataset.feature_std[i]:12.4f} "
            f"{abs_err[:, i].max():12.3e} {rel_err[:, i].max():12.3e}"
        )
    print(f"\nworst relative error: {rel_err.max():.3e}")

    test_standardization_roundtrip(seed=seed)
    print("test_standardization_roundtrip: OK")
