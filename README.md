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