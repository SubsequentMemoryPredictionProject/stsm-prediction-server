
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from DB import dbAccess
from DB.dbAccess import get_data
import config as cfg


elc1 = [1, 1, 1, 1, 1]
elc2 = [1, 1, 1, 1, 1]
elc3 = [1, 1, 1, 1, 1]
elc4 = [1, 1, 1, 1, 1]
elc5 = [1, 1, 1, 1, 1]

elc6 = [2, 2, 2, 2, 2]
elc7 = [2, 2, 2, 2, 2]
elc8 = [2, 2, 2, 2, 2]
elc9 = [2, 2, 2, 2, 2]
elc10 = [2, 2, 2, 2, 2]

def get_features():
    get_data(cfg.mysql)

features = elc1 + elc2 + elc3 + elc4 + elc5
results = [1, 1, 1]
features2 = elc6 + elc7 + elc8 + elc9 + elc10
results2 = [2, 2, 2]

featuresArray = np.asarray([features,features2])
resultsArray = np.asarray([results, results2])

print(featuresArray)
print(resultsArray)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(featuresArray,resultsArray)

#save model
joblib.dump(knn, 'savedModel.pkl')






