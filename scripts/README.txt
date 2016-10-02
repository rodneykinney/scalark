PREPARING THE DATA

Training data must be in tab-separated format. The first line should be a header containing column names.
All columns whose names do not begin with the '#' character will be interpreted as real-valued features.

TRAINING A RANKER

Use the train-ranker.sh script to train a ranking model. In addition to the numerical features,
the training data file must have a column that is a query ID and a column that is a relevance label.
The names of these columns can be specified on the command line. They default to "#Query" and "#Label".
The location of the model file (in json format) is specified via the -output parameter. Finally, specify
the order, from worst to best, of the values that will appear in the label column, e.g "b,g,e".
The trainer does not make use of NDCG weights for the labels, only their relative order.

EVALUATING A RANKER

Use the score-rows.sh script to assign model scores to a data set. Give the location of a trained model
(i.e. the json file produced by the train-ranker.sh script) via the -model parameter. The output will
be the same as the input, but with an additional column appended containing the model score.

IMPORTANT: The order of features must be the same in the evaluation dataset as it was in the training set.
Trees refer to features by index, not by name, so if the columns are in different order
in the train and test set, the model will not score correctly.
