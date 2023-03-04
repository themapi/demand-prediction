from __future__ import annotations

import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor


class RandomForestForecastModel:
    """
    RandomForest model to forecast 24 h ahead energy demand.
    """

    def __init__(self, **model_hparams):
        """
        Initialize RandomForest model.
        Args:
            **model_hparams(dict): Dictionary containing `sklearn.ensemble.RandomForestRegressor` constructor params.
        """
        self.model_hparams = model_hparams
        self._rf_model = RandomForestRegressor(**model_hparams)

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Train the model on the features and target.

        Args:
            features(pd.DataFrame): Training input samples.
            target(pd.DataFrame): Training target samples

        Returns:
            None

        """

        self._rf_model.fit(features, target)

    def forecast(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Forecast target based on input.

        Args:
            features: input features.

        Returns:
            pd.DataFrame: 24 h ahead prediction for each input sample.
        """
        return self._rf_model.predict(features)

    def save(self, path: str) -> None:
        """
          Persist model to local drive.

          Args:
              path(str): Path to file.

          Returns:
              None
          """
        dump(self, path)

    @classmethod
    def load(cls, path: str) -> RandomForestForecastModel:
        """
        Load model from local drive.

        Args:
            path: Path to serialized model file.

        Returns:
            RandomForestForecastModel: deserialized model.
        """
        model = load(path)

        if not isinstance(model, RandomForestForecastModel):
            raise ValueError('Path points to wrong model type.')

        return model
