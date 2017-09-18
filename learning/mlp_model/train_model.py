from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
import pymysql.connections
import sys
import os
import json

PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
import config as cfg

from DB.db_access import get_features
from DB.db_access import get_results
from model_evaluation.test_model import evaluate_model



conn = pymysql.connect(host=cfg.mysql['host'], passwd=cfg.mysql['password']
                     , port=cfg.mysql['port'], user=cfg.mysql['user'], db=cfg.mysql['database'])

mlp_model = MLPClassifier(max_iter=400)

# load data to train & test model
features = get_features(conn)
results = get_results(conn)

conn.close()
# split data to training and testing set
features_train, features_test, \
    results_train, results_test = train_test_split(features, results, test_size=0.25, random_state=0)

multi_mlp_model = MultiOutputClassifier(mlp_model, n_jobs=1)
multi_mlp_model.fit(features_train, results_train)

predictions = multi_mlp_model.predict(features_test)


evaluate_model(multi_mlp_model,features_test,results_test)
# save trained model
joblib.dump(multi_mlp_model, 'mlp_model.pkl')
