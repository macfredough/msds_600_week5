import pandas as pd
import timeit 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('prepped_churn_data_913222.csv', index_col='customerID')
df.rename(columns = {'TotalCharges_tenure ratio':'charge_per_tenure'}, inplace=True)

newchurndf = pd.read_csv('new_churn_data.csv', index_col='customerID')
#rename Churn column to target to work in function
df.rename(columns = {'Churn':'target'}, inplace = True)
#set function for pipeline for training set and test set
def TPOT_Pipeline(trainingdata, testdata):
    features = trainingdata.drop('target', axis=1)
    targets = trainingdata['target']
    
    tpot_data = trainingdata.copy()
    
    training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

    # Average CV score on the training set was: 0.8039435021253766
    exported_pipeline = RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.25, min_samples_leaf=12, min_samples_split=5, n_estimators=100)
    # Fix random state in exported estimator
    if hasattr(exported_pipeline, 'random_state'):
        setattr(exported_pipeline, 'random_state', 42)

    exported_pipeline.fit(training_features, training_target)
    results = exported_pipeline.predict(testdata)
    print(results)
    
    testing_features = testdata
    for row in range(len(testing_features)):
        resultsproba = exported_pipeline.predict_proba(testing_features)
        prob = (resultsproba[row][0]*100).round(decimals=1)
        print({testing_features.index[row]}, {prob})

TPOT_Pipeline(df, newchurndf)