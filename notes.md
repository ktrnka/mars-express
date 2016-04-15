Issues
======

* Communication issues may result in gaps in the data
* Usually one observation is made every 30 seconds or 60 seconds. The goal of the competition is to predict the average electric current per hour.
* Umbra and penumbra events are typically around 30 minutes long but we're predicting in 1-hour intervals so we need to represent them differently.

Evaluation notes
----------------

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

Modeling notes
==============

Predict mean is around 0.125.

Linear regression works well without tuning. Bagged linear regression is a big improvement over that. Random forests
appear to work well on training data but doesn't generalize to the testing data. Gradient boosting is even worse than
RF. Running adaboost on linear regression did very poorly. Neural networks appear to generalize extremely well but
have NaN problems, likely due to training for mse/mae.

Linear regression notes
-Sometimes when I add rolling means it overfits and has enormous MSE like 10k. When I sampled the rolling means as ints
that fixed it. This problem was even more apparent with some of the LR variants like PassiveAggressiveRegressor and LARS

Bagged linear regression notes
-sklearn bagging wrapper doesn't support multivariate. Initially I used the multivariate wrapper around bagging around LR
but it was slow. About 10x speedup by writing my own multivariate bagging wrapper.
-At first I would set a fixed number of features but when using PCA that broke cause the features were reduced so now
I'm lazy and just say 80% or so.
-Number of samples isn't that useful; tuning prefers high values close to 100%. It'd probably prefer 100% but I'm using
a distribution in the random seach
-Bumping up from 10 to 30 estimators didn't do much but seems a little more stable
-Overall it fluctuates a bit
-Tried a variation where I train the first 1/3 of estimators then compute feature importances then use those feature
importances to weight the feature sampling in the remaining 2/3. This was consistently worse even when I tried things
other than mean of the per-estimator importances. There were many variations because I have a feature importance vector
per estimator and per output. So I tried a mean over the outputs and min over the estimators and also tried max over outputs
then min over estaimtors. I filled in unknown ones with the overall mean. None of them really helped. I made sure I was taking
absolute value of the coef_ array and everything. Also I found that numpy has a numpy.random.choice that supports a probability
weighting.

Random forest notes
-Does worse with PCA, closer to test with PCA
-In every single test the test error was way high, like over 2 stddev away from the CV error. Was worse than linear regression usually
-generally it wanted shorter trees without too many features. number of trees didn't matter much

Gradient boosting notes
-Needed to use with multivariate wrapper so it was slow af
-Fits training data better than RF, fits testing data worse than RF

AdaBoost
-Couldn't get any improvement over bagging

Neural network
-More input noise is generally better, even with PCA
-NaN continues to be a big problem
-Activation function unclear but elu/relu tend to NaN more, sigmoid was preferred with PCA
-Learning rate is very important even with Adam
-TimeCV almost always leads to NaN at certain resolutions
-PCA with high dimensions can lead to NaN
-W + b maxnorm constraint fixed NaN in one case (tried it 4x in a row and it was fixed, but didn't fix all cases)
-Batch norm triggers NaN on even the first epoch. W constraint doesn't fix it nor does disabling dropout
-Small L2 on bias = NaN immediately
-Early stopping helps prevent NaN but hurts test acc usually even with like 20 epochs patience
-NaN happens whether I use Adam, RMSProp, or SGD/momentum
-Adam higher eps doesn't help
-Reducing number of outputs didn't help
-NaN in test data seemed to be happening on rows where sy was way out of the regular range so I started using the ClippedRobustScaler for that
-Tried to save Keras weights to file as it trained but it requires h5py which requires a binary external that didn't install
-Doing a StandardScaler after RobustScaler doesn't fix it
-Feature selection doesn't fix NaN
-Hard coding FTL features instead of automatically selecting them didn't help

Gaussian process
-I tried it but then it was using 37gb ram and wasn't done so I had to kill it.

Fit a linear + cos curve on time alone
-Didn't beat linear regression on time alone = there may not be seasonal trends at a fixed period

Evaluation metrics and CV
=========================
Default is over the martian years which is basically 3-fold without shuffle. This correlates fairly well with test except for tree-based models.

Setting up cross-val to always extrapolate ran into some interesting design decisions about time slice window, gaps, 
extrapolate to earlier times, etc.

When predicting test data it's useful to check the MAE vs predict-mean as a sanity check.

sklearn issues
==============
Multivariate wrapper is a pain in the ass because of the way it interacts with random search. For gradient boosting for instance, normally
it looks like:
MultivariateRegressionWrapper(GradientBoostingRegressor(...))

But for random search I need to do this:
MultivariateRegressionWrapper(RandomizedSearchCV(GradientBoostingRegressor(), gb_hyperparams, ...)

So then I can't actually summarize the hyperparams in the usual way.


Feature engineering
===================

Tried automated search over log/sqrt/sin/cos/square, feeding the single feature into linear regression and checking
cross-validated MSE vs the untransformed features. log transforms on some features were useful.

Found a gradient method in numpy which just does local numeric derivative. Very useful for some features at least according to the test.

Tried automated search over rolling means. This found tons of potential features but it was specific to the CV split I'm using.
The resulting features definitely helped linear regression but didn't help NN as much. The odd thing about it is that I already
had some rolling features like say event_counts, event_counts_2h, event_counts_5h. It wanted a rolling mean of 50 on all of them
which seems odd. I'd think it would want a particular trigger distance.

Tried automated search over multiplied/divided/substracted features but only found one potential one. When I added it, it led
to great overfitting.

Sampling is a bit confusing. For instance, the target is hourly so I resample the power data to hourly. But then to map
other features to that I need to do reindex which only has a "nearest" option not a distance-weighted option. So I've tried doing
things like resampling saaf data to 30 minutes before reindexing to match the power data. It doesn't seem to matter though.

Also resampling the longterm data to a sub-day index then interpolating doesn't seem to help much even though it's the right thing to do.
 
Mapping events to times was tricky. Like say we have an event from 1:03pm to 2:50pm but the main index is hourly. We could
set an indicator at 1pm to 1 and 0 otherwise but that doesn't really convey it. At first I tried diffing and assigning
the amount of time but then I found a MUCH easier way:
-Generate a 5-minute sampled index and assign indicator variables.
-Rolling mean to convert to hourly
-This gives a nice smooth integral of time spent in a given state over the last hour/day/etc
-It took some refactoring but coding this up functionally was much much cleaner.

Phobos and Deimos umbra/penumbra events are very short (like 2 min tops) and infrequent so I excluded them.

At first I tried to make the penumbra generator exclude the umbra times but really it's only in penumbra but not umbra for
a minute or so tops. So I dropped that.

Simple sum of event/dmop counts in an hourly rolling fashion was easy to code, useful for the models, and very robust.




Ideas
-----
var_period_amplitude * sin(days_in_space / var_period + var_period_offset) + var_linear_amplitude * days_in_space + var_log_amplitude * log(days_in_space) + var_base

Data notes
==========

Output value distributions
--------------------------
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

Average intervals
-----------------
```
Data set	Average time interval
0	Power (resampled)	00:59:59.927219
1	Long term	23:59:28.558951
2	Event	00:12:01.389474
3	DMOP	00:06:08.299687
4	SAAF	00:00:51.110708
5	FTL	00:59:07.295944
```

Average event durations
-----------------------
```
[('MAR_PENUMBRA', Timedelta('0 days 00:31:04.897512')),
 ('MAR_UMBRA', Timedelta('0 days 00:30:50.016860')),
 ('DEI_PENUMBRA', Timedelta('0 days 00:02:05.720000')),
 ('DEI_UMBRA', Timedelta('0 days 00:00:44')),
 ('PHO_PENUMBRA', Timedelta('0 days 00:00:42.714285')),
 ('PHO_UMBRA', Timedelta('0 days 00:00:15.300000'))]
```

FTL events (incl test)
----------------------
```
SLEW 27317
EARTH 20420
INERTIAL 7154
D4PNPO 3409
MAINTENANCE 2653
NADIR 2173
WARMUP 1261
ACROSS_TRACK 1033
RADIO_SCIENCE 690
D1PVMC 404
D9PSPO 148
D2PLND 114
SPECULAR 47
NADIR_LANDER 42
D3POCM 22
D7PLTS 16
SPOT 15
D8PLTP 6
D5PPHB 4
```

```
	flagcomms	flagcomms_60min	SLEW	SLEW_60m	EARTH	EARTH_60m	INERTIAL	INERTIAL_60m	D4PNPO	D4PNPO_60m	...	ACROSS_TRACK	ACROSS_TRACK_60m	RADIO_SCIENCE	RADIO_SCIENCE_60m	D1PVMC	D1PVMC_60m	D9PSPO	D9PSPO_60m	D2PLND	D2PLND_60m
count	593557.000000	593546.000000	593557.000000	593546.000000	593557.000000	593546.000000	593557.000000	593546.000000	593557.000000	593546.000000	...	593557.000000	593546.000000	593557.000000	593546.000000	593557.000000	593546.000000	593557.000000	593546.000000	593557.000000	593546.000000
mean	0.215922	0.215925	0.148481	0.148484	0.571851	0.571844	0.060237	0.060238	0.058326	0.058327	...	0.016047	0.016048	0.027896	0.027897	0.003991	0.003991	0.000886	0.000886	0.000352	0.000352
std	0.411461	0.377646	0.355577	0.195262	0.494811	0.419785	0.237925	0.171508	0.234360	0.196271	...	0.125658	0.101239	0.164676	0.150614	0.063050	0.043681	0.029756	0.024341	0.018761	0.012361
min	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	...	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
25%	0.000000	0.000000	0.000000	0.000000	0.000000	0.083333	0.000000	0.000000	0.000000	0.000000	...	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
50%	0.000000	0.000000	0.000000	0.000000	1.000000	0.666667	0.000000	0.000000	0.000000	0.000000	...	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
75%	0.000000	0.333333	0.000000	0.333333	1.000000	1.000000	0.000000	0.000000	0.000000	0.000000	...	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
max	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	...	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000
```

Top 10 DMOP
AACFE91A     37317
AACFE03A     36424
AAAAF56A1    20035
APSF28A1     18312
APSF38A1     16881
AACFE05A     16719
AAAAF20E1    13440
AMMMF10A0    11852
AVVV02A0     10073
AVVV03B0      9774
