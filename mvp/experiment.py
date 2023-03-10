import os

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit

from mvp.data_processing import load_training_data
from mvp.regression_model import RandomForestForecastModel

EXPERIMENTS_PATH = 'experiments'


class Experiment:

    def __init__(self, experiments_path=EXPERIMENTS_PATH, cv_spilits=3, n_lag_days=2, model_hparams=None):
        self.cv_spilits = cv_spilits
        self.n_lag_days = n_lag_days

        self.experiments_path = experiments_path

        if model_hparams is None:
            model_hparams = dict()
        self.model = RandomForestForecastModel(**model_hparams)

        # create experiments path is not existing.
        os.makedirs(self.experiments_path, exist_ok=True)

        self.run = self._determine_run()

        self.current_run_path = os.path.join(self.experiments_path, f'run_{self.run:03d}')
        os.makedirs(self.current_run_path)

    def _determine_run(self):
        experiments = os.listdir(self.experiments_path)
        if len(experiments) == 0:
            return 0

        versions = list(map(lambda e: int(e.split('_')[-1]), experiments))
        return max(versions) + 1

    def exec(self):
        # TODO replace console logs with proper logging

        print(f'Starting run {self.run}.', self.current_run_path)

        print('Train and save model')
        self.train()

        print('Evaluate model')
        eval_res = self.evaluate()
        print('evaluation result')
        print(eval_res)

    def train(self):
        """
        Train the model.

        Returns:

        """
        x, y = load_training_data(self.n_lag_days)
        self.model.train(x, y)
        model_path = os.path.join(self.current_run_path, f'model.joblib')
        self.model.save(model_path)
        print('saved model here:', model_path)

    def evaluate(self) -> pd.DataFrame:
        """
        Perform model evaluation.

        Returns:
            pd.DataFrame:

        """
        x, y = load_training_data(self.n_lag_days)

        tscv = TimeSeriesSplit(n_splits=self.cv_spilits)

        score_list = list()
        step_name_list = list()
        fold_list = list()

        for fold, (train_idx, val_idx) in enumerate(tscv.split(x, y)):
            x_train = x.iloc[train_idx]
            y_train = y.iloc[train_idx]

            x_val = x.iloc[val_idx]
            y_val = y.iloc[val_idx]

            # create new model instance and train
            model = self.model.__class__(**self.model.model_hparams)
            model.train(x_train, y_train)

            r2_train = metrics.r2_score(y_train, model.forecast(x_train))
            score_list.append(r2_train)
            step_name_list.append('train')
            fold_list.append(fold)

            r2_val = metrics.r2_score(y_val, model.forecast(x_val))
            score_list.append(r2_val)
            step_name_list.append('val')
            fold_list.append(fold)

        csv_path = os.path.join(self.current_run_path, 'evaluate.csv')
        result = pd.DataFrame({'scores': score_list, 'step': step_name_list, 'fold': fold_list})
        result.to_csv(csv_path)
        return result
