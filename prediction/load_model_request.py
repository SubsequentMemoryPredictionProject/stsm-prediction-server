from sklearn.externals import joblib
from DB.db_access import get_signals
import os
import sys

PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)

def load_model():
    mlp_model = joblib.load('mlp_model.pkl')
    return mlp_model


def prediction_request(request):
    prediction_details = "AND user_id=" + str(request[0]) + " AND subject_id=" + str(request[1])\
                         + " AND word_id=" + str(request[2])
    user_request = get_signals(prediction_details)
    return user_request

req = [1, 2, 50]
model = load_model()
features_to_predict = prediction_request(req)
for i in features_to_predict:
    print(i)
prediction = model.predict(features_to_predict)
print(prediction)