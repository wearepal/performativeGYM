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

    2. **Feature standardization**:
       Features are normalized to zero mean and unit variance:

       .. math::

           x \\leftarrow \\frac{x - \\mu}{\\sigma}.

    3. **Class balancing**:
       The dataset is subsampled to obtain equal numbers of positive and negative
       examples:

       .. math::

           |\\{i : y_i = 1\\}| = |\\{i : y_i = 0\\}|.

    4. **Shuffling**:
       The dataset is randomly permuted using a JAX PRNG key.

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

    def _load_data(self, seed: int) -> tuple[npt.NDArray, npt.NDArray, list[str]]:
        key = jax.random.PRNGKey(seed)

        data = pd.read_csv(self.datapath, index_col=0)
        data.dropna(inplace=True)

        features = data.drop("SeriousDlqin2yrs", axis=1)
        feature_names = list(features.columns)
        # zero mean, unit variance
        features = self.standard_scaler.fit_transform(features)

        outcomes = np.array(data["SeriousDlqin2yrs"])  # 120000 samples

        # balance classes
        default_indices = np.where((outcomes == 1))[0]  # 8000
        other_indices = np.where((outcomes == 0))[0][: len(default_indices)]  # 112000
        indices = np.concatenate((default_indices, other_indices))

        features_balanced = features[indices]
        outcomes_balanced = outcomes[indices]

        # shuffle arrays
        shuffled = jax.random.permutation(key, len(indices))
        return features_balanced[shuffled], outcomes_balanced[shuffled], feature_names

    def __len__(self):
        return len(self.labels)
