"""Load and preprocess Kaggle credit dataset."""

import os

import jax
import numpy as np
from numpy import typing as npt
import pandas as pd  # type: ignore
from sklearn import preprocessing  # type: ignore

class CreditDataset:
    """Class to lazily load the credit dataset."""

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
        outcomes = np.array(data["SeriousDlqin2yrs"]) #120000 samples

        # balance classes
        default_indices = np.where((outcomes == 1))[0] #8000
        other_indices = np.where((outcomes == 0))[0][:len(default_indices)] # 112000
        indices = np.concatenate((default_indices, other_indices))
        #indices = np.arange(outcomes.shape[0])

        features_balanced = features[indices]
        outcomes_balanced = outcomes[indices]

        shape = features_balanced.shape

        # shuffle arrays
        shuffled = jax.random.permutation(key, len(indices))
        return features_balanced[shuffled], outcomes_balanced[shuffled]

    def __len__(self):
        return len(self.labels)