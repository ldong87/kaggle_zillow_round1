# kaggle_zillow_round1

This repo documents my model for the Zillow competition round 1 on Kaggle and summarizes lessons learned.

data_transform.py cleans data.

data_alloc.py and data_alloc2.py allocates data for first and second level models, respectively.

model1*.py and model2*.py are first and second level models, respectively.

gp_feat.py recreates features from https://www.kaggle.com/scirpus/genetic-programming-lb-0-0643904 for 2017 data.

I considered weak temporal pattern in this data. Used forward-chaining cross validation. For example, train on Quarter1 and validate on Quarter2, then train on Q1+Q2 and validate on Q3, etc. Since there are two years data and the last quarter is to be predicted, we have 7 quarters data available. Forward-chaining cross validation training strategy is used for the first 6 quarters. The 7th quarter is reserved for final stacking. This pipeline is improved by parallelizing among CV folds and automatic hyperparameter selection in the kaggle_porto_seguro repo. This strategy itself gave top 10% ranking. By mixing with my teammates awesome results, our ranking bumped to top 1%.

Here are the libraries used in this pipeline: Scikit learn, Catboost, LightGB, Vowpal Wabbit, XGB.
