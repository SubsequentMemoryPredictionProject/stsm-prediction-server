from sklearn.externals import joblib
from learning.data_set import get_features
import os
import sys

PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)

def load_model():
    mlp_model = joblib.load('mlp_model.pkl')
    return mlp_model


def prediction_request(request):
    prediction_details = "user_id=" + str(request[0]) + " AND subject_id=" + str(request[1])\
                         + " AND word_id=" + str(request[2])
    query = "SELECT signal_elec1_subelec1, signal_elec1_subelec2," \
            " signal_elec1_subelec3, signal_elec2_subelec1, signal_elec2_subelec2," \
            " signal_elec2_subelec3, signal_elec3_subelec1, signal_elec3_subelec2," \
            " signal_elec3_subelec3, signal_elec4_subelec1, signal_elec4_subelec2," \
            " signal_elec4_subelec3 FROM data_set WHERE " + prediction_details
    user_request = get_features(query)
    return user_request

req = [1, 2, 50]
model = load_model()
features_to_predict = prediction_request(req)
for i in features_to_predict:
    print(i)
prediction = model.predict(features_to_predict)
print(prediction)