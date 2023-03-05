# Challenge forecast next 24 hours Energy demand

Build a minimal viable product (MVP) that forecasts the next 24h of energy demand in the state of Victoria (AUS). The
dataset is located [**here**](https://raw.githubusercontent.com/JoaquinAmatRodrigo/skforecast/master/data/vic_elec.csv).

The CSV time-series dataset is in **30-min resolution**, spans from **2012-01-01 to 2014-12-31** totaling in
**52,608 entries**. It contains following features:

- **Time** (datetime): UTC timestamp of the entry.
- **Demand** (float): energy demand in Mega Watts (MW)
- **Temperature** (float):  temperature in [I suppose] degrees Celsius
- **Date** (date): local date as.
- **Holiday** (binary): flag indicate a holiday.

The MVP should contain following steps:

- [x] load data
- [x] data processing/feature extraction
- [x] train model to predict 24 h energy demand
- [x] evaluate model performance
- [x] expose model via REST API

Later steps:

- [ ] data versioning
- [ ] data validation
- [ ] check for drifts
- [ ] log metrics
- [ ] log artifacts
- [ ] hyperparameter tuning
- [ ] model versioning

## Model and Preprocessing

We use the classic **Random Forest (RF) regressor** to forecast the upcoming 24h energy demand.
Unlike many other regression models RF models are not distance based and therefore do not need any feature
normalization/scaling.

We extract following features:

| Feature Name         | Type        | Description                                                               |
|----------------------|-------------|---------------------------------------------------------------------------|
| Weekday              | int         | Day of the week of the sample                                             |
| Weekend              | bool        | Flag indicating sample records a weekend                                  |
| DayOfYear            | int         | Day of year ranging from 1 to 365 in a normal year and 366 in a leap year |
| Hour_UTC             | int         | UTC hour of the day                                                       |
| Historic_Demand      | list[float] | Demand of the previous **_n_** days                                       |
| Historic_Temperature | list[float] | Temperature of the previous **_n_** days                                  |
| Holiday_Lagged       | list[bool]  | Holiday flag of the previous **_n_** days                                 |
| Holiday_target       | list[bool]  | Holiday flag of the upcoming 24 hours                                     |

Note that _n_ is a hyperparameter that need to be optimized.

## Setup Environment

We use conda to manage the projects' environment.
Before getting started install and setup conda.

Execute following command to create the development environment:

```$ conda env create -f environment.yml```

Next activate the created environment:

```$ conda activate demand-prediction```

## Model training and evaluation

After setting up and activating the conda environment we can train and evaluate the model.
Run ``$ python python main_cli.py run_exp`` inside th project root to do so.

With the default setting the model performs good (r2>0.8) but clearly overfits.
In future iterations hyperparameters need to be optimized.

## Start REST server and API usage

The REST server can be started as easy as training is done.

``$ python main_cli.py start_rest --model_path=<path to model>``

**NOTE**: server is stil at development stage and has not been tested yet. 