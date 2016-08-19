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

### Ensembling

* Simple average works
* Really promising results at the end by feeding outputs from multiple models into ElasticNet on the last 10% of data (the data used for early stopping in NN/RNN) and running hyperparam optimization. But I goofed when including the random forest and forgot to train it only on 90% so it botched my very last submission. The weirdest thing about it is that for 4 models it's 4 * 33 features and 33 outputs. So the model can learn to cross info from different power lines if need be. Anyway that was the easiest way to set it up.

## Other

* Keras 1.0.1 gave worse MSE than 1.0.0 despite the speedup so I reverted. Took a while to figure that out tho.
* Learned Amazon EC2 and GPU. Cnmem super important. Use fast math for GPU. Also Keras unroll=True speedup for RNN. GPU maybe 10x for MLP but only like 2x for RNN. The speed tests are in [this doc](analysis/GPU speed testing.csv). Also did some [basic matrix mult benchmarks on CPU](analysis/numpy benchmarks.csv).
