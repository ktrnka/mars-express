Mars Express Power Challenge
======

This repo covers about half of my code for competitng in ESA's [Mars Express machine learning competition](https://kelvins.esa.int/mars-express-power-challenge). The other half is covered by a git submodule in .gitmodules so if you clone this repo, clone with --recursive.

Code organization
----
Almost everything is in src/.

* train_test.py: This is what I was using for a long time to run cross-validated tests of various models. In older versions the data loading was here too, so if you want to check the exact features around the time of my best solution it's in this file. There's also some hyperparameter tuning when enabled.
* run_experiment.py: This because a dumping ground for one-off experiments like ensembling 2 clones of the same RNN with different random init, random subspace methods on top of neural networks, weighting later data more strongly, early stopping tests, L2 tests, RNN with ReLU output, approximated bidirectional RNN, stateful RNN, etc.
* run_feature_selection.py: The home of mega feature selection experiments. In previous versions the data loader that adds about 1000 features was here. I'm in the process of migrating feature selection methods into the shared library.
* run_predict_test.py: Train a model on training data and generate test set predictions along with some graphs.
* train_clipper.py: The neural network style models would sometimes make crazy predictions outside of the training data range of values, leading to higher error. This just checks the range of values for each output in the training data and saves a json file with those.
* loaders.py: Towards the end of the competition I pulled all the data loading and feature engineering code into this file.

What worked and what didn't
----
I'll do some writeups on [my blog](https://kwtrnka.wordpress.com/) about this but haven't to it yet. A lot of my experiments are in trello_archive.json.