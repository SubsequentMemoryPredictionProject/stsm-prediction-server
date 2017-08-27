from sklearn.externals import joblib
import numpy as np

knn = joblib.load('C:\\Users\\user\PycharmProjects\stsm-prediction-server\learning\savedModel.pkl')
newWord = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
newWordNumpy = np.asarray([newWord])
prediction = knn.predict(newWordNumpy)
print(prediction)