Issues
======

* Communication issues may result in gaps in the data
* Usually one observation is made every 30 seconds or 60 seconds. The goal of the competition is to predict the average electric current per hour.
* Umbra and penumbra events are typically around 30 minutes long but we're predicting in 1-hour intervals so we need to represent them differently.

Evaluation notes
----------------

* Some subsystems use more power than others:
```
    [('NPWD2532', 1.2784544539416436),
     ('NPWD2451', 0.70638696960115044),
     ('NPWD2562', 0.41776833851835621),
     ('NPWD2551', 0.3514824066928644),
     ('NPWD2561', 0.32679764486951718),
     ('NPWD2851', 0.29735586700095001),
     ('NPWD2491', 0.20062931770559825),
     ('NPWD2771', 0.18702746065757272),
     ('NPWD2402', 0.17533859711581995),
     ('NPWD2722', 0.17143021160408523),
     ('NPWD2721', 0.13688889013260905),
     ('NPWD2802', 0.13341522482676968),
     ('NPWD2372', 0.12657178867261701),
     ('NPWD2791', 0.10721812037715052),
     ('NPWD2881', 0.044849927342612769),
     ('NPWD2531', 0.043983705005978126),
     ('NPWD2742', 0.013405408220475326),
     ('NPWD2821', 0.0057802114337378386),
     ('NPWD2501', 0.0052953547221308503),
     ('NPWD2552', 0.003969890813022319),
     ('NPWD2882', 0.003373378050076258),
     ('NPWD2481', 0.0024368076938512655),
     ('NPWD2482', 0.0020357552727119934),
     ('NPWD2401', 0.0016735955305568957),
     ('NPWD2801', 0.0015355098237033397),
     ('NPWD2472', 0.0012366238401949506),
     ('NPWD2792', 0.00098645796607530899),
     ('NPWD2872', 0.00074503690100553475),
     ('NPWD2471', 0.00073860585143870072),
     ('NPWD2692', 0.00057839801825718823),
     ('NPWD2852', 0.0005276393334864805),
     ('NPWD2871', 0.00052165085541985509),
     ('NPWD2691', 2.3118022691713985e-06)]
```
* Some of the signals stay mostly at their mean with little variation. Others vary over time.

Baselines
---------

* DummyRegressor(mean): 0.264 +/- 0.039
* Although there's typically only 2-3 values, predict mean is much better than predict median.
* Upper bound if we know the outcome in various frequency bins on the raw data without resampling first:
```
RMS with 7D approximation: 0.213
RMS with 1D approximation: 0.211
RMS with 12H approximation: 0.210
RMS with 6H approximation: 0.207
RMS with 2H approximation: 0.199
RMS with 1H approximation: 0.192
RMS with 30M approximation: 0.234
```

Model types
-----------
Linear regression works quite well without any tuning. Random forest regression only works well when I run it through a randomzied search on the hyperparams.

With all output classes:
```
LinearRegression: 0.113 +/- 0.011
RandomForestRegression: 0.112 +/- 0.008
RandomForestRegression(tuned): 0.107 +/- 0.009
```

Ideas
-----
var_period_amplitude * sin(days_in_space / var_period + var_period_offset) + var_linear_amplitude * days_in_space + var_log_amplitude * log(days_in_space) + var_base