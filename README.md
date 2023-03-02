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

- [ ] load data
- [ ] data processing/feature extraction
- [ ] train [any] model to predict 24 h energy demand
- [ ] evaluate model performance
- [ ] expose model via REST API

Later steps:
- [ ] data versioning
- [ ] data validation
- [ ] check for drifts
- [ ] log metrics
- [ ] log artifacts
- [ ] hyperparameter tuning
- [ ] model versioning

## Model training and evaluation
TODO

## start REST server and API usage
TODO

