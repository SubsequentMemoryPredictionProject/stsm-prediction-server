#import mysql.connector
import pymysql.connections
from sklearn.preprocessing import Imputer  # for fixing incomplete data
import numpy as np
NUM_FEATURES = 532
NUM_ELECTRODES = 6


# get data from DB
def get_data(db, query):
    cursor = db.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    print("in db access")
    cursor.close()
    return data


# insert data to DB
def insert_data(db, query):
    cursor = db.cursor()
    cursor.execute(query)
    cursor.close()
    db.commit()
    return


# get eeg signals from DB
def get_signals(db):
    print('in get signals')
    signals = []
    word = []
    query1 = 'SELECT signal_elec1_subelec1, signal_elec1_subelec2, \
             signal_elec1_subelec3, signal_elec2_subelec1, signal_elec2_subelec2, \
             signal_elec2_subelec3 from data_set WHERE EEG_data_section=1'
    query2 = 'SELECT signal_elec3_subelec1, signal_elec3_subelec2, \
             signal_elec3_subelec3, signal_elec4_subelec1, signal_elec4_subelec2, \
             signal_elec4_subelec3 FROM data_set WHERE EEG_data_section=2 '
    section_one = get_data(db, query1)
    print("got section one")
    print(len(section_one))
    section_two = get_data(db, query2)
    print("got section 2")
    for i in range(len(section_one)):
        print("word  = ",i)
        for j in range(NUM_ELECTRODES):
            word.extend(float_arr(section_one[i][j]))
        for k in range(NUM_ELECTRODES):
            word.extend(float_arr(section_two[i][k]))
        signals.append(np.asarray(word, dtype=np.float16))
        word = []
    return signals


# create float array from str data  + fix missing signals
def float_arr(string):
    fix = False
    to_array = string.split(',')
    for i in range(len(to_array)):
        # ignore missing words
        if 'undefined' == to_array[i]:
            to_array = np.zeros(NUM_FEATURES,np.float16)
            return to_array
        # mark the places with missing signals
        if to_array[i] in ['', '.', '-', ' ']:
            to_array[i] = np.nan
            fix = True
            continue
        to_array[i] = np.float16(to_array[i])
    # add place holders for missing signals if array contains < NUM_FEATURES
    while len(to_array) < NUM_FEATURES:
        to_array.append(np.nan)
        fix = True
    if fix:
        to_array = fix_missing_signals(to_array)
    return to_array


# get results from DB
def get_results(db):
    print('in get results')
    results = []
    query = 'SELECT stm, stm_confidence_level, stm_remember_know, ltm, \
             ltm_confidence_level, ltm_remember_know FROM data_set WHERE EEG_data_section=1 '
    data_set = get_data(db, query)
    for row in data_set:
        # ignore missing words
        #if row[1] == 0 or row[4] == 0:
           # print("no results")
            #continue
        results.append(np.array(row, int))
    return results


# fix missing signals from one electrode
def fix_missing_signals(electrode):
    imp = Imputer(axis=1, copy=False, missing_values='NaN', strategy='mean', verbose=0)
    imp.fit([electrode])
    return np.reshape(imp.transform([electrode]), NUM_FEATURES)
