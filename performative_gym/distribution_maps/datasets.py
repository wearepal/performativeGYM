"""Load and preprocess Kaggle credit dataset."""

import os

import jax
import numpy as np
from numpy import typing as npt
import pandas as pd
from sklearn import preprocessing


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

    3. **Bias augmentation**:
       A constant feature equal to 1 is appended to each data point:

       .. math::

           x \\leftarrow (x, 1),

       which allows linear models to incorporate an intercept term.

    4. **Class balancing**:
       The dataset is subsampled to obtain equal numbers of positive and negative
       examples:

       .. math::

           |\\{i : y_i = 1\\}| = |\\{i : y_i = 0\\}|.

    5. **Shuffling**:
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
        Preprocessed feature matrix, including the bias term.

    labels : ndarray of shape (n,)
        Binary labels indicating default status.

    Properties
    ----------
    num_agents : int
        Number of samples in the dataset:

        .. math::

            n = \\text{features.shape}[0].

    num_features : int
        Number of features per sample (including the bias term):

        .. math::

            d = \\text{features.shape}[1].

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
    - The added bias feature simplifies the use of linear models by avoiding the
      need for a separate intercept parameter.
    - Shuffling is performed using JAX for consistency with the rest of the codebase.
    """

    def __init__(self, datafile: str = "credit_data.zip", *, seed: int):
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        datapath = os.path.join(cur_dir, datafile)

        self.datapath = datapath
        self.seed = seed
        self.features, self.labels = self.load_data(self.seed)

    @property
    def num_agents(self):
        """Compute number of agents in the dataset."""
        return self.features.shape[0]

    @property
    def num_features(self):
        """Compute number of features for each agent."""
        return self.features.shape[1]

    def load_data(self, seed: int) -> tuple[npt.NDArray, npt.NDArray]:
        key = jax.random.PRNGKey(seed)

        data = pd.read_csv(self.datapath, index_col=0)
        data.dropna(inplace=True)

        features = data.drop("SeriousDlqin2yrs", axis=1)
        # zero mean, unit variance
        features = preprocessing.scale(features)

        # add bias term
        features = np.append(features, np.ones((features.shape[0], 1)), axis=1)
        outcomes = np.array(data["SeriousDlqin2yrs"])  # 120000 samples

        # balance classes
        default_indices = np.where((outcomes == 1))[0]  # 8000
        other_indices = np.where((outcomes == 0))[0][: len(default_indices)]  # 112000
        indices = np.concatenate((default_indices, other_indices))

        features_balanced = features[indices]
        outcomes_balanced = outcomes[indices]

        # shuffle arrays
        shuffled = jax.random.permutation(key, len(indices))
        return features_balanced[shuffled], outcomes_balanced[shuffled]

    def __len__(self):
        return len(self.labels)
