from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn import svm
import pymysql.connections
import config as cfg
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from DB.db_access import get_signals
from DB.db_access import get_results
from model_evaluation.test_model import evaluate_model


conn = pymysql.connect(host=cfg.mysql['host'], passwd=cfg.mysql['password']
                     , port=cfg.mysql['port'], user=cfg.mysql['user'], db=cfg.mysql['database'])

svm_model = svm.LinearSVC()
# load data to train & test model
X = get_signals(conn)
Y = get_results(conn)
conn.close()

print(np.shape(X))
print(np.shape(Y))
# split data to training and testing set
X_train, X_test, \
    Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

multi_svm_model = MultiOutputClassifier(svm_model, n_jobs=1)
multi_svm_model.fit(X_train, Y_train)

evaluate_model(multi_svm_model, X_test, Y_test)

# save trained model
joblib.dump(multi_svm_model, 'svm_model.pkl')
