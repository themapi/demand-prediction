from typing import Optional, Tuple

import numpy as np
import pandas as pd

DATA_SOURCE = 'https://raw.githubusercontent.com/JoaquinAmatRodrigo/skforecast/master/data/vic_elec.csv'


def load_training_data(n_lag_days: int):
    """
    Load data into memory and extract
    Args:
        n_lag_days: see `extract_training_data_from`.

    Returns:
        Tuple: Training data tuple: `(features, target)`.
    """
    data = _load_train_data()
    return process_data(data, n_lag_days)


def process_data(data, n_lag_days):
    data = resample_data(data)
    data = extract_training_data_from(data, n_lag_days)
    return drop_na(*data)


def drop_na(target, features):
    drop_idx = target.isna().any(axis=1) | features.isna().any(axis=1)
    drop_idx = drop_idx[drop_idx].index

    features = features.drop(drop_idx)
    target = target.drop(drop_idx)
    return target, features


def extract_training_data_from(data: pd.DataFrame, n_lag_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract features and target from dataset.

    Args:
        data: resampled dataset with 1H resolution.
        n_lag_days: `Demand`, `Temperature`, and `Holiday` of `n_day_days` previous to the sample are extracted.

    Returns:
        Tuple: Training data tuple: `(features, target)`.
    """
    # holiday flags of the upcoming 24 hours
    holiday_future = _shift_vars(data, 'Holiday', np.arange(1, 25) * -1, 'Target-Holiday')

    # past `n_lag_days` of Demand, Temperature and Holiday
    demand_lagged = _shift_vars(data, 'Demand', np.arange(24 * n_lag_days))
    temperature_lagged = _shift_vars(data, 'Temperature', np.arange(24 * n_lag_days))
    holiday_lagged = _shift_vars(data, 'Holiday', np.arange(24 * n_lag_days))

    data['DST'] = _extract_dst(data)
    data['Weekday'] = data['Date'].dt.weekday
    data['Weekend'] = data['Weekday'].isin([6, 7])
    data['DayOfYear'] = data['Date'].dt.dayofyear
    data['TomorrowIsWeekend'] = data['Weekday'].isin([5, 6])
    data['Hour_UTC'] = data.index.hour

    normal_feats = data[['Weekday', 'Weekend', 'Hour_UTC', 'DayOfYear']]
    features = pd.concat([normal_feats, holiday_future, demand_lagged, temperature_lagged, holiday_lagged], axis=1)

    target = _shift_vars(data, 'Demand', np.arange(1, 25) * -1)

    return features, target


def resample_data(data):
    return data.set_index('Time').resample('1H', label='left').mean(numeric_only=False)


def _load_train_data():
    # load data into memory
    return pd.read_csv(DATA_SOURCE, sep=',', parse_dates=['Time', 'Date'],
                       dtype={'Demand': 'float32', 'Temperature': 'float32'})


def _shift_vars(df: pd.DataFrame, var_name: str, hour_shift_list: np.ndarray,
                res_column_prefix: Optional[str] = None) -> pd.DataFrame:
    """
    Shift multiple variables either by positive or negative period.

    Shift variables is based on `pandas.DataFrame.shift`.

    Args:
        df: the dataframe.
        var_name: the column to shift.
        hour_shift_list: an array of positive or negative hourly to shift.
        res_column_prefix: The prefixes of the lagged column. Default `var_name`.

    Returns:
        pd.DataFrame: lagged variable.
     """
    shifts = list()

    df = df[[var_name]]

    if res_column_prefix is None:
        res_column_prefix = var_name

    def shift_symbol(period):
        return '' if period < 0 else '+'

    for hour_shift in hour_shift_list:
        shift = df[var_name].shift(periods=hour_shift)
        shift.name = f'{res_column_prefix}{shift_symbol(hour_shift)}{hour_shift:03d}'
        shifts.append(shift)
    return pd.concat(shifts, axis=1)


def _extract_dst(data):
    local_time = data.index.to_series().dt.tz_convert(tz='Australia/Victoria')
    # 1 if DST, 0 otherwise.
    return local_time.apply(lambda ts: ts.dst().total_seconds() // (60 * 60))
