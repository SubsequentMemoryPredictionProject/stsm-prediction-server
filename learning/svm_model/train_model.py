from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn import svm
import pymysql.connections
import config as cfg
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from DB.db_access import get_features
from DB.db_access import get_results


conn = pymysql.connect(host=cfg.mysql['host'], passwd=cfg.mysql['password']
                     , port=cfg.mysql['port'], user=cfg.mysql['user'], db=cfg.mysql['database'])

svm_model = svm.LinearSVC()
# load data to train & test model
features = get_features(conn)
results = get_results(conn)
conn.close()

# split data to training and testing set
features_train, features_test, \
    results_train, results_test = train_test_split(features, results, test_size=0.25, random_state=0)

multi_svm_model = MultiOutputClassifier(svm_model, n_jobs=1)
multi_svm_model.fit(features_train, results_train)

predictions = multi_svm_model.predict(features_test)
for row in predictions:
    print(row)
# save trained model
joblib.dump(multi_svm_model, 'svm_model.pkl')
