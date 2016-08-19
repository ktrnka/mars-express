# Mars Express Power Challenge

This repo covers about half of my code for competitng in ESA's [Mars Express machine learning competition](https://kelvins.esa.int/mars-express-power-challenge). The other half is covered by a git submodule in .gitmodules so if you clone this repo, clone with --recursive.

## Code organization
Almost everything is in src/.

* train_test.py: This is what I was using for a long time to run cross-validated tests of various models. In older versions the data loading was here too, so if you want to check the exact features around the time of my best solution it's in this file. There's also some hyperparameter tuning when enabled.
* run_experiment.py: This because a dumping ground for one-off experiments like ensembling 2 clones of the same RNN with different random init, random subspace methods on top of neural networks, weighting later data more strongly, early stopping tests, L2 tests, RNN with ReLU output, approximated bidirectional RNN, stateful RNN, etc.
* run_feature_selection.py: The home of mega feature selection experiments. In previous versions the data loader that adds about 1000 features was here. I'm in the process of migrating feature selection methods into the shared library.
* run_predict_test.py: Train a model on training data and generate test set predictions along with some graphs.
* train_clipper.py: The neural network style models would sometimes make crazy predictions outside of the training data range of values, leading to higher error. This just checks the range of values for each output in the training data and saves a json file with those.
* loaders.py: Towards the end of the competition I pulled all the data loading and feature engineering code into this file.

## What worked and what didn't
I'll do some writeups on [my blog](https://kwtrnka.wordpress.com/) about this but haven't to it yet. A lot of my experiments are in trello_archive.json.

# Notes for writeup

Right now I'm trying to get the breadth of interesting experiments and I'll merge in results as I go through Trello. I started off recording experiments in google sheets but I didn't like the organization of it. Now I'm realizing it's a huge pain to archive the hyperparameter experimental notes which I put in sheets notes on the result values and that doesn't export... but for now here's the [raw doc](analysis/experimental results.csv).

## Feature engineering

* Representing umbra and other begin/end events as time spent in that range by setting indicators on 5-min windows then integrating
* Bad initial design doing things like 1h, 2h, 5h time lags. When doing the massive feature selection experiment I found it was much cleaner to do powers of 4 rather than anything like powers of 2 because when you do powers of 4 the various time delays are only 25% overlapped not 50%. Feature correlation was a huge issue in the mega FS experiment.
* sin/cos of angles (no improvement)
* feature combinations with plus/minus/mult/div (no improvement, but this was early on). There's some results of this on daily sampled data in [feature_interactions.txt](analysis/feature_interactions.txt)
* automatically tried a bunch of feature delays early on in ipython, that found some decent features like that nadir 400h one. Also automatically tried sqrt/log. Did this all by training linear regression on the base feature in isolation and then the transformed version then measuring improvement/degradation. Got some decent ones here. 
* wanted to get things like percent of time sx was near 30 deg but I couldn't figure out how to automatically find the relevant reference points
* SAAF rolling histogram was moderately useful, took some pandas tricks to get it working
* Discrete cosine transform and fourier transform in feature selection experiment. Only one of the DCT features made it into top 100 so I gave up (sx dct component 1 I think).
* Tried correcting for latency on DMOP and FTL assuming that they were the earth-send timestamps not received. Slightly helpful for DMOP, harmful for FTL. Also tried the reverse for EVTF but that was harmful.
* Relative altitude was slightly helpful
* For subsystems, tried deltas between all pairs of subsys commands (no improvement)

### Mega feature selection

* Leave one out is great when you have a small set of non-overlapping features. It's amazingly horrible though when it's a big set of features like 1000
* Tried weighting for diversity by penalizing all but the highest ranked variant of a base feature (usually was harmful in CV)
* 

## Hyperparameter tuning

* Extended sklearn's RandomSearchCV to run statistical tests on each tuned hyperparameter (assuming they're independent). Pearson for continuous and Anova for categorical. Note that Pearson has issues if it's a bowl shape so I tried "folding" the curve over the max but this never found anything different so I removed the code.
* I had to do some annoying stuff to analyse hyperparam tuning that was inside the multivariate wrapper I used to get univariate gradient boosting to work. But I abandoned it cause I couldn't get GB to do well.

## Debugging

* Simple check for crazy predictions was percent deviation from predict-mean, caught some errors early especially predictions with NaN
* When doing cross-validation, save mean and stddev. Then compute the z-score of leaderboard for the appropriate model to interpret.

## Models

* Bagged linear regression worked pretty well though not as well as ElasticNet. On one hand it was silly that I didn't realize EN was there. On the other hand now I have a general random subspace wrapper so I got to try that even with the neural networks.
* Fitting time-based functions with scipy.optimize was horrible. Tried linear + periodic components of time but it was awful.
* Random forest and gradient boosting did okay on training data early on but very bad z-score on leaderboard (much worse than expected). Did the usual hyperparam tuning too.
* Tried AdaBoost with linear regression to try and get something like gradient boosted linear regression rather than trees. Was slow and performed badly.
* Gaussian processes (the ram usage..... never again....)
* Tried a wrapper that predicted the hourly mean and hourly deltas then ensembled extrapolations (horrible failure)

### Neural network/MLP

* Had tons of NaN problems early on. Never entirely clear what fixed them except maybe reducing batch size and a small about of L2. High amount of L2 even caused NaN somehow. It was very unsatisfying cause I ran dozens of tests and it wasn't entirely clear what fixed it. Maybe a bad Theano version for all I know.
* Batch normalization caused NaN 100% of time, gave up on that
* Small amount of gaussian input noise helpful (results in google docs)
* When using PCA, sigmoid was best activation
* When not using PCA, ELU > ReLU > sigmoid/tanh
* Even with Adam optimizer, decaying the learning rate sped up convergence and improved stability. Used 0.99 exponential decay or thereabouts.
* Like I've had before, early stopping is nice but you have to set the patience in Keras to some number of epochs. I did something like 10% of the max epochs.
* Like any gradient method, sensitive to the scaling of features. Normally use StandardScaler but I found there were tons of outliers with that so switched to RobustScaler. Still many outliers so clipped the values from RobustScaler at -5 to 5.
* Tried sample_weights to have it fit more recent data more strongly. Inconclusive results but I didn't test on leaderboard.
* From looking at test set graphs, learned that MLP could predict negative values which is total nonsense for power. At first I used an output augmentation wrapper to clip the values to positive. Then I learned the actual min/max values per line from the raw (not hourly) data and used those for clipping. Then I realized that actually that gives almost no gain over just making sure they're positive. Only after all that crap did I remember that ReLU is max(0, x) which does exactly what I need. Big improvement from having ReLU output units in CV tests. Didn't try on leaderboard I don't think.
* Tried the subspace wrapper with neural nets, small improvement
* At the end I did a sort of annealing process inspired by a reddit comment and I didn't really need to tune it and it gave better results than exponential decay learning rate.
* Some tests around wider MLPs and adding a layer of depth in conjunction with learning rate.

Test 1

```
Validation score -0.0881 +/- 0.0090, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (800,), u'nn__estimator__input_noise': 0.03598438240339102, u'nn__estimator__input_dropout': 0.02, u'nn__estimator__learning_rate': 0.00048626015800653599}
Validation score -0.0882 +/- 0.0087, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (400,), u'nn__estimator__input_noise': 0.03769585034600211, u'nn__estimator__input_dropout': 0.01, u'nn__estimator__learning_rate': 0.00015922827933410926}
Validation score -0.0883 +/- 0.0088, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (400,), u'nn__estimator__input_noise': 0.04121338193692153, u'nn__estimator__input_dropout': 0.01, u'nn__estimator__learning_rate': 0.00010974987654930576}
Validation score -0.0885 +/- 0.0090, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (800,), u'nn__estimator__input_noise': 0.043880816415799426, u'nn__estimator__input_dropout': 0.01, u'nn__estimator__learning_rate': 0.00050941380148163823}
Validation score -0.0885 +/- 0.0084, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (800,), u'nn__estimator__input_noise': 0.017764275864457224, u'nn__estimator__input_dropout': 0.01, u'nn__estimator__learning_rate': 0.00011497569953977378}
Validation score -0.0887 +/- 0.0085, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (100,), u'nn__estimator__input_noise': 0.03568816446004866, u'nn__estimator__input_dropout': 0.01, u'nn__estimator__learning_rate': 0.00017475284000076848}
Validation score -0.0888 +/- 0.0091, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (800,), u'nn__estimator__input_noise': 0.039177693026757096, u'nn__estimator__input_dropout': 0, u'nn__estimator__learning_rate': 0.0010235310218990265}
Validation score -0.0889 +/- 0.0089, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (800,), u'nn__estimator__input_noise': 0.037007559926665705, u'nn__estimator__input_dropout': 0, u'nn__estimator__learning_rate': 0.00038535285937105326}
Validation score -0.0893 +/- 0.0089, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (100,), u'nn__estimator__input_noise': 0.03866624997739228, u'nn__estimator__input_dropout': 0.02, u'nn__estimator__learning_rate': 0.00012045035402587825}
Validation score -0.0895 +/- 0.0084, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (100, 100), u'nn__estimator__input_noise': 0.010618474524977502, u'nn__estimator__input_dropout': 0, u'nn__estimator__learning_rate': 0.0013530477745798072}
Validation score -0.0895 +/- 0.0095, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (200,), u'nn__estimator__input_noise': 0.015695170649236062, u'nn__estimator__input_dropout': 0.02, u'nn__estimator__learning_rate': 0.001072267222010324}
Validation score -0.0895 +/- 0.0088, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (100,), u'nn__estimator__input_noise': 0.040678938170706086, u'nn__estimator__input_dropout': 0, u'nn__estimator__learning_rate': 0.0035938136638046297}
Validation score -0.0902 +/- 0.0097, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (200, 200), u'nn__estimator__input_noise': 0.02970225654574587, u'nn__estimator__input_dropout': 0.01, u'nn__estimator__learning_rate': 0.0028480358684358047}
Validation score -0.0903 +/- 0.0092, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (100,), u'nn__estimator__input_noise': 0.025263249875233103, u'nn__estimator__input_dropout': 0.01, u'nn__estimator__learning_rate': 0.0029836472402833404}
Validation score -0.0904 +/- 0.0096, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (100,), u'nn__estimator__input_noise': 0.015194731574771674, u'nn__estimator__input_dropout': 0.01, u'nn__estimator__learning_rate': 0.010000000000000004}
Validation score -0.0905 +/- 0.0088, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (200, 200), u'nn__estimator__input_noise': 0.0355250265970331, u'nn__estimator__input_dropout': 0.02, u'nn__estimator__learning_rate': 0.00025353644939701147}
Validation score -0.0905 +/- 0.0082, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (200, 200), u'nn__estimator__input_noise': 0.0397982449902545, u'nn__estimator__input_dropout': 0.02, u'nn__estimator__learning_rate': 0.00011497569953977378}
Validation score -0.0906 +/- 0.0092, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (200, 200), u'nn__estimator__input_noise': 0.026233124665614234, u'nn__estimator__input_dropout': 0.02, u'nn__estimator__learning_rate': 0.00035111917342151353}
Validation score -0.0906 +/- 0.0099, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (100, 100), u'nn__estimator__input_noise': 0.037268767133084954, u'nn__estimator__input_dropout': 0.02, u'nn__estimator__learning_rate': 0.0065793322465756846}
Validation score -0.0907 +/- 0.0087, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (200, 200), u'nn__estimator__input_noise': 0.018327900767854503, u'nn__estimator__input_dropout': 0, u'nn__estimator__learning_rate': 0.00077426368268112742}
Hyperparameter correlations with evaluation metric
        nn__estimator__learning_rate: Pearson r = -0.4504, p = 0.0463
        nn__estimator__hidden_layer_sizes: Anova f = 11.1736, p = 0.0002
                (400,): -0.0883
                (800,): -0.0885
                (200,): -0.0895
                (100,): -0.0897
                (100, 100): -0.0900
                (200, 200): -0.0905
        nn__estimator__input_noise: Pearson r = 0.3379, p = 0.1451
        nn__estimator__input_dropout: Pearson r = -0.1838, p = 0.4380
experiment_neural_network took 2.0 hours, 17.7 minutes
```

Test 2 with a narrower range of learning rate

```
Validation score -0.0876 +/- 0.0085, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (800,), u'nn__estimator__input_noise': 0.020930883900154396, u'nn__estimator__input_dropout': 0.01, u'nn__estimator__learning_rate': 0.00025950242113997342}
Validation score -0.0877 +/- 0.0085, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (400,), u'nn__estimator__input_noise': 0.04216356972440097, u'nn__estimator__input_dropout': 0.01, u'nn__estimator__learning_rate': 0.00015556761439304716}
Validation score -0.0879 +/- 0.0084, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (800,), u'nn__estimator__input_noise': 0.023498597770852322, u'nn__estimator__input_dropout': 0.01, u'nn__estimator__learning_rate': 8.9021508544503871e-05}
Validation score -0.0880 +/- 0.0089, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (800,), u'nn__estimator__input_noise': 0.03737057468742709, u'nn__estimator__input_dropout': 0.01, u'nn__estimator__learning_rate': 0.00095454845666183405}
Validation score -0.0880 +/- 0.0086, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (200,), u'nn__estimator__input_noise': 0.04569973153781531, u'nn__estimator__input_dropout': 0.01, u'nn__estimator__learning_rate': 0.00086974900261778355}
Validation score -0.0880 +/- 0.0084, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (400,), u'nn__estimator__input_noise': 0.029157773023451176, u'nn__estimator__input_dropout': 0.01, u'nn__estimator__learning_rate': 0.00086974900261778355}
Validation score -0.0881 +/- 0.0086, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (200,), u'nn__estimator__input_noise': 0.03863325014228302, u'nn__estimator__input_dropout': 0.01, u'nn__estimator__learning_rate': 0.0004750810162102794}
Validation score -0.0881 +/- 0.0082, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (200,), u'nn__estimator__input_noise': 0.040245442868896544, u'nn__estimator__input_dropout': 0.01, u'nn__estimator__learning_rate': 0.00011768119524349986}
Validation score -0.0882 +/- 0.0086, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (200,), u'nn__estimator__input_noise': 0.02946351399401527, u'nn__estimator__input_dropout': 0.01, u'nn__estimator__learning_rate': 0.00015556761439304716}
Validation score -0.0884 +/- 0.0086, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (800,), u'nn__estimator__input_noise': 0.03975464086053151, u'nn__estimator__input_dropout': 0, u'nn__estimator__learning_rate': 0.00065793322465756824}
Validation score -0.0890 +/- 0.0081, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (400,), u'nn__estimator__input_noise': 0.02701798870232455, u'nn__estimator__input_dropout': 0, u'nn__estimator__learning_rate': 7.3907220335257725e-05}
Validation score -0.0902 +/- 0.0082, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (200,), u'nn__estimator__input_noise': 0.036204441041047225, u'nn__estimator__input_dropout': 0, u'nn__estimator__learning_rate': 4.2292428743894961e-05}
Validation score -0.0905 +/- 0.0081, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (400,), u'nn__estimator__input_noise': 0.020489044495970478, u'nn__estimator__input_dropout': 0, u'nn__estimator__learning_rate': 2.6560877829466868e-05}
Validation score -0.0909 +/- 0.0088, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (50, 50), u'nn__estimator__input_noise': 0.025094836956684997, u'nn__estimator__input_dropout': 0.01, u'nn__estimator__learning_rate': 0.00054622772176843423}
Validation score -0.0913 +/- 0.0088, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (50, 50), u'nn__estimator__input_noise': 0.044914732624225184, u'nn__estimator__input_dropout': 0.01, u'nn__estimator__learning_rate': 0.00017886495290574361}
Validation score -0.0917 +/- 0.0082, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (200,), u'nn__estimator__input_noise': 0.027169429275428564, u'nn__estimator__input_dropout': 0.01, u'nn__estimator__learning_rate': 1.5922827933410907e-05}
Validation score -0.0918 +/- 0.0083, Hyperparams {u'nn__estimator__hidden_units': None, u'nn__estimator__hidden_layer_sizes': (800,), u'nn__estimator__input_noise': 0.021036283638904786, u'nn__estimator__input_dropout': 0, u'nn__estimator__learning_rate': 1.7475284000076831e-05}
```

### Recurrent neural networks

(starts around gdoc line 416)

* Tried LSTM and GRU. Couldn't get reasonable performance out of GRU. Only somewhat close to LSTM when I doubled the number of units in GRU.
* About half of the lessons from MLPs were invalid for RNNs.
* Decaying the learning rate was harmful for LSTM. No clue why. I tried several methods of doing it and none worked but I may have tested badly.
* For MLP I could try out general ideas against daily averages of power. For RNN it really doesn't work on those so there isn't an equivalent fast-test and that really killed iteration speed.
* Tried with 4, 8, 12 time steps. 4 time steps seemed best.
* Early stopping like with MLP helped speed it up and prevent overfitting. But it requires ~10% held-out data which is the most recent 10%. Tried what I called post-training which was to refit the model for just a few epochs on 100% of data after early stopping triggered on the 90/10 split. In retrospect this may have helped a bit.
* Simple ensemble of clones that I pulled from the Google neural machine translation paper. It's a shame that it works.
* Recurrent dropout was important
* In retrospect it seems like more input noise helped it for leaderboard.
* ReLU output units should've worked but they were always equal or worse than letting the network predict negative and the clipping as post-processing. I'm still stumped about this and maybe it's really that I needed to retune learning rate.
* Tried running without early stopping and the results were all over the place. Randomness killed that test.

### Ensembling

* Simple average works
* Really promising results at the end by feeding outputs from multiple models into ElasticNet on the last 10% of data (the data used for early stopping in NN/RNN) and running hyperparam optimization. But I goofed when including the random forest and forgot to train it only on 90% so it botched my very last submission. The weirdest thing about it is that for 4 models it's 4 * 33 features and 33 outputs. So the model can learn to cross info from different power lines if need be. Anyway that was the easiest way to set it up.

I did a lot of experiments around this basic sort of stacking and I'll try to copy most of the experiments. Test on daily samples:

```
Base models:
ValidationWrapper(Pipeline(ElasticNet)): 0.0718 +/- 0.0159
ValidationWrapper(Pipeline(RidgeCV)): 0.0788 +/- 0.0177
OutputTransformation(Pipeline(OutputTransformation(Nn))): 0.0536 +/- 0.0043

Ridge stacking(ValidationWrapper(Pipeline(ElasticNet)),ValidationWrapper(Pipeline(RidgeCV)),OutputTransformation(Pipeline(OutputTransformation(Nn))))
StackedEnsemble: 0.0490 +/- 0.0135
RidgeCV stacking(ValidationWrapper(Pipeline(ElasticNet)),ValidationWrapper(Pipeline(RidgeCV)),OutputTransformation(Pipeline(OutputTransformation(Nn))))
StackedEnsemble: 0.0549 +/- 0.0200

ElasticNet stacking(ValidationWrapper(Pipeline(ElasticNet)),ValidationWrapper(Pipeline(RidgeCV)),OutputTransformation(Pipeline(OutputTransformation(Nn))))
StackedEnsemble: 0.0598 +/- 0.0171

MLP stacking(ValidationWrapper(Pipeline(ElasticNet)),ValidationWrapper(Pipeline(RidgeCV)),OutputTransformation(Pipeline(OutputTransformation(Nn))))
StackedEnsemble: 0.0716 +/- 0.0156
```

Strangely my default settings for RidgeCV were best. Tuning the regularization hurt things. ElasticNet wasn't great. MLP was AWFUL. Tried on daily samples:

```
Base models:
ValidationWrapper(Pipeline(ElasticNet)): 0.0993 +/- 0.0054
ValidationWrapper(Pipeline(RidgeCV)): 0.1001 +/- 0.0094
OutputTransformation(Pipeline(OutputTransformation(Nn))): 0.0857 +/- 0.0117

Ridge stacking(ValidationWrapper(Pipeline(ElasticNet)),ValidationWrapper(Pipeline(RidgeCV)),OutputTransformation(Pipeline(OutputTransformation(Nn))))
StackedEnsemble: 0.0871 +/- 0.0107
```

So even though I introduced two crappy models and it's L2 norm it doesn't hurt RMS much and improved stddev of RMS. Back to daily samples with some hyperparam tuning inside the stacked model:

```
ValidationWrapper(Pipeline(ElasticNet)): 0.0718 +/- 0.0159
ValidationWrapper(Pipeline(RidgeCV)): 0.0788 +/- 0.0177
OutputTransformation(Pipeline(OutputTransformation(Nn))): 0.0545 +/- 0.0051

Ridge stacking(ValidationWrapper(Pipeline(ElasticNet)),ValidationWrapper(Pipeline(RidgeCV)),OutputTransformation(Pipeline(OutputTransformation(Nn))))
StackedEnsemble: 0.0483 +/- 0.0125
Ridge/tuned stacking(ValidationWrapper(Pipeline(ElasticNet)),ValidationWrapper(Pipeline(RidgeCV)),OutputTransformation(Pipeline(OutputTransformation(Nn))))
StackedEnsemble: 0.0501 +/- 0.0112

StackedEnsemble: 0.0596 +/- 0.0172
ElasticNet/tuned stacking(ValidationWrapper(Pipeline(ElasticNet)),ValidationWrapper(Pipeline(RidgeCV)),OutputTransformation(Pipeline(OutputTransformation(Nn))))
StackedEnsemble: 0.0506 +/- 0.0136

MLP stacking(ValidationWrapper(Pipeline(ElasticNet)),ValidationWrapper(Pipeline(RidgeCV)),OutputTransformation(Pipeline(OutputTransformation(Nn))))
StackedEnsemble: 0.0651 +/- 0.0219
StackedEnsemble: 0.0530 +/- 0.0182
```

Ran some tests overnight on daily samples. I did the simple base ensemble of ridge/elasticnet/MLP and then a secoond one that was forwards RNN, backwards RNN, and MLP.

```
ValidationWrapper(Pipeline(ElasticNet)): 0.0993 +/- 0.0054
ValidationWrapper(Pipeline(RidgeCV)): 0.1001 +/- 0.0094
OutputTransformation(Pipeline(OutputTransformation(Nn))): 0.0848 +/- 0.0117

Ridge stacking(ValidationWrapper(Pipeline(ElasticNet)),ValidationWrapper(Pipeline(RidgeCV)),OutputTransformation(Pipeline(OutputTransformation(Nn))))
StackedEnsemble: 0.0892 +/- 0.0099
Ridge/tuned stacking(ValidationWrapper(Pipeline(ElasticNet)),ValidationWrapper(Pipeline(RidgeCV)),OutputTransformation(Pipeline(OutputTransformation(Nn))))
StackedEnsemble: 0.0890 +/- 0.0119

ElasticNet stacking(ValidationWrapper(Pipeline(ElasticNet)),ValidationWrapper(Pipeline(RidgeCV)),OutputTransformation(Pipeline(OutputTransformation(Nn))))
StackedEnsemble: 0.0857 +/- 0.0113
ElasticNet/tuned stacking(ValidationWrapper(Pipeline(ElasticNet)),ValidationWrapper(Pipeline(RidgeCV)),OutputTransformation(Pipeline(OutputTransformation(Nn))))
StackedEnsemble: 0.0845 +/- 0.0099

MLP stacking(ValidationWrapper(Pipeline(ElasticNet)),ValidationWrapper(Pipeline(RidgeCV)),OutputTransformation(Pipeline(OutputTransformation(Nn))))
StackedEnsemble: 0.1064 +/- 0.0172
StackedEnsemble: 0.0918 +/- 0.0125


OutputTransformation(Pipeline(OutputTransformation(Rnn))): 0.0799 +/- 0.0111
OutputTransformation(Pipeline(OutputTransformation(Rnn))): 0.1545 +/- 0.0096
OutputTransformation(Pipeline(OutputTransformation(Nn))): 0.0853 +/- 0.0094

Ridge stacking(OutputTransformation(Pipeline(OutputTransformation(Rnn))),OutputTransformation(Pipeline(OutputTransformation(Rnn))),OutputTransformation(Pipeline(OutputTransformation(Nn))))
StackedEnsemble: 0.0791 +/- 0.0098

Ridge/tuned stacking(OutputTransformation(Pipeline(OutputTransformation(Rnn))),OutputTransformation(Pipeline(OutputTransformation(Rnn))),OutputTransformation(Pipeline(OutputTransformation(Nn))))
StackedEnsemble: 0.0802 +/- 0.0110
```

Even though the backwards RNN is crap and the MLP isn't nearly as good as RNN the plain Ridge ensemble shows lower RMS and lower stddev. Note though that I had just made a change that broke the backwards RNN; usually it wasn't THAT bad. Retested with the backwards RNN fixed and a clone of the forwards RNN.

```
Base models:
OutputTransformation(Pipeline(OutputTransformation(Rnn))): 0.0787 +/- 0.0092
OutputTransformation(Pipeline(OutputTransformation(Rnn))): 0.0810 +/- 0.0069
ValidationWrapper(OutputTransformation(Pipeline(OutputTransformation(Rnn)))): 0.0939 +/- 0.0058
Pipeline(OutputTransformation(Nn)): 0.0848 +/- 0.0114

Ridge stacking
StackedEnsemble: 0.0767 +/- 0.0098
Ridge/tuned stacking
StackedEnsemble: 0.0790 +/- 0.0110

ElasticNet stacking
StackedEnsemble: 0.0769 +/- 0.0110
ElasticNet/tuned stacking
StackedEnsemble: 0.0752 +/- 0.0106
```

Both the ridge stacking and tuned ElasticNet stacking are considerably better than any individual model. For elasticnet I'm tuning not just the alpha but also the L1/L2 ratio from 0.25 to 0.75. The nice thing is that Ridge and ElasticNet are very fast so I can afford to run 100 iterations of RandomSearchCV.

### Gradient boosting

Towards the end I came back to gradient boosting and tried the setup Alex mentioned: convert to univariate by indicator features for the output to predict. At first I had issues because you need to stride the data a certain way to get KFold cross val to not hold out specific outputs.

Promising test on daily samples. Before tuning it's not much different than just using the multivariate wrapper but after tuning it's a bit better.

```
JoinedMultivariateWrapper(GradientBoosting): 0.0541 +/- 0.0148
Validation score -0.0670 +/- 0.0136, Hyperparams {u'max_features': 86, u'min_samples_split': 4, u'learning_rate': 0.08421161308119386, u'max_depth': 8, u'subsample': 0.9}
Validation score -0.0684 +/- 0.0139, Hyperparams {u'max_features': 88, u'min_samples_split': 36, u'learning_rate': 0.10726376084269512, u'max_depth': 8, u'subsample': 0.9}
Validation score -0.0694 +/- 0.0151, Hyperparams {u'max_features': 102, u'min_samples_split': 10, u'learning_rate': 0.20692082142894658, u'max_depth': 7, u'subsample': 0.9}
Validation score -0.0701 +/- 0.0120, Hyperparams {u'max_features': 87, u'min_samples_split': 3, u'learning_rate': 0.36472359230935886, u'max_depth': 7, u'subsample': 0.9}
Validation score -0.0704 +/- 0.0138, Hyperparams {u'max_features': 86, u'min_samples_split': 6, u'learning_rate': 0.1663921101291721, u'max_depth': 8, u'subsample': 0.9}
Validation score -0.0709 +/- 0.0175, Hyperparams {u'max_features': 91, u'min_samples_split': 12, u'learning_rate': 0.059621935911239224, u'max_depth': 10, u'subsample': 0.9}
Validation score -0.0727 +/- 0.0142, Hyperparams {u'max_features': 92, u'min_samples_split': 35, u'learning_rate': 0.39011463294387844, u'max_depth': 9, u'subsample': 0.9}
Validation score -0.0728 +/- 0.0144, Hyperparams {u'max_features': 87, u'min_samples_split': 11, u'learning_rate': 0.33760095579949395, u'max_depth': 8, u'subsample': 0.9}
Validation score -0.0761 +/- 0.0184, Hyperparams {u'max_features': 106, u'min_samples_split': 33, u'learning_rate': 0.3865466379856397, u'max_depth': 10, u'subsample': 0.9}
Validation score -0.0773 +/- 0.0203, Hyperparams {u'max_features': 96, u'min_samples_split': 4, u'learning_rate': 0.30805294742920464, u'max_depth': 11, u'subsample': 0.9}
Hyperparameter correlations with evaluation metric
        max_features: Pearson r = -0.5458, p = 0.1027
        min_samples_split: Pearson r = -0.1501, p = 0.6789
        learning_rate: Pearson r = -0.6729, p = 0.0330
        max_depth: Pearson r = -0.7554, p = 0.0115
JoinedMultivariateWrapper(RandomizedSearchCV(GradientBoosting)): 0.0510 +/- 0.0138
test_new_gradient_boosting took 58.5 minutes

MultivariateWrapper(GradientBoosting): 0.0534 +/- 0.0134
Best hyperparameters for grid search inside of multivariate regression
        max_features: 93.48 +/- 7.24
        min_samples_split: 21.12 +/- 12.93
        learning_rate: 0.29 +/- 0.12
        max_depth: 3.73 +/- 0.83
        subsample: 0.90 +/- 0.00
MultivariateWrapper(RandomizedSearchCV(GradientBoosting)): 0.0581 +/- 0.0155
tune_gradient_boosting took 6.0 minutes
```

But it's very slow. The tuning actually didn't finish on daily samples after 2 days.

## Other

* Keras 1.0.1 gave worse MSE than 1.0.0 despite the speedup so I reverted. Took a while to figure that out tho.
* Learned Amazon EC2 and GPU. Cnmem super important. Use fast math for GPU. Also Keras unroll=True speedup for RNN. GPU maybe 10x for MLP but only like 2x for RNN. The speed tests are in [this doc](analysis/GPU speed testing.csv). Also did some [basic matrix mult benchmarks on CPU](analysis/numpy benchmarks.csv).
* For both kinds of neural networks I was using gaussian input noise to help prevent overfitting. But then I realized that skewing the LTDATA by up to 5% was silly cause it's so reliable so I came up with time based noise. I'd do a random weighted average in a window around each time. At first I did this with a nice matrix mult but then learned that a 50000x50000 matrix isn't a good way to go when it's pretty sparse and I don't know how to use numpy sparse matrices too well. In the end I didn't get to use it for training but used it for feature selection to identify features that fluctuated too much