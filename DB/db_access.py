#import mysql.connector
import pymysql.connections
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
    print('in get data')
    signals = []
    word = []
    query1 = 'SELECT signal_elec1_subelec1, signal_elec1_subelec2, \
             signal_elec1_subelec3, signal_elec2_subelec1, signal_elec2_subelec2, \
             signal_elec2_subelec3 from data_set WHERE EEG_data_section=1 LIMIT 0,4000'
    query2 = 'SELECT signal_elec3_subelec1, signal_elec3_subelec2, \
             signal_elec3_subelec3, signal_elec4_subelec1, signal_elec4_subelec2, \
             signal_elec4_subelec3 FROM data_set WHERE EEG_data_section=2 LIMIT 0,4000'
    section_one = get_data(db, query1)
    print("got section one")
    print(len(section_one))
    section_two = get_data(db, query2)
    print("got section 2")
    for i in range(0, len(section_one)):
        print("word  = ",i)
        for j in range(NUM_ELECTRODES):
            word.extend(float_arr(section_one[i][j]))
        for k in range(NUM_ELECTRODES):
            word.extend(float_arr(section_two[i][k]))
        word = np.asarray(word, dtype=np.float16)
        signals.append(word)
        word = []
    #signals = np.asarray(signals,dtype=np.ndarray)
    return signals


# create float array from str
def float_arr(string):
    to_array = string.split(',')
    for i in range(len(to_array)):
        # ignore missing words
        if 'undefined' == to_array[i]:
            to_array = np.zeros(532,np.float16)
            return to_array
        # TODO fix missing features
        if (to_array[i] == '') or (to_array[i] == '-') or (to_array[i] == '.'):
            to_array[i]= np.float16(0.0)
            continue
        to_array[i] = np.float16(to_array[i])
    # add missing feature
    while len(to_array) < NUM_FEATURES:
        to_array.append(np.float16(0.0))
    return to_array


# get results from DB
def get_results(db):
    print('in get results')
    results = []
    query = 'SELECT stm, stm_confidence_level, stm_remember_know, ltm, \
             ltm_confidence_level, ltm_remember_know FROM data_set WHERE EEG_data_section=1 LIMIT 0,4000'
    data_set = get_data(db, query)
    for row in data_set:
        # ignore missing words
        #if row[1] == 0 or row[4] == 0:
           # print("no results")
            #continue
        results.append(np.array(row, int))
    results = np.asarray(results,int)
    return results