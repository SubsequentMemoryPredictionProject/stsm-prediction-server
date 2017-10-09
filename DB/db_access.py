#import mysql.connector
import pymysql.connections
from sklearn.preprocessing import Imputer  # for fixing incomplete data
import numpy as np
import sys, os
PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
from logger import Logger
NUM_FEATURES = 532
NUM_ELECTRODES = 6


logger = Logger().get_logger()


# get data from DB
def get_data(db, query):
    logger.info('DB access - in get data')
    cursor = db.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    return data


# insert data to DB
def insert_data(db, query):
    logger.info('DB access - in insert data')
    cursor = db.cursor()
    cursor.execute(query)
    cursor.close()
    db.commit()
    return


# get eeg signals from DB
def get_signals(db, user_query='', table='data_set'):
    logger.info('In get signals: using table - %s' % table)
    signals = []
    word = []
    if user_query:
        user_query = 'AND ' + user_query
    query1 = 'SELECT signal_elec1_subelec1, signal_elec1_subelec2, \
             signal_elec1_subelec3, signal_elec2_subelec1, signal_elec2_subelec2, \
             signal_elec2_subelec3 FROM ' + table + ' WHERE EEG_data_section=1 ' + user_query + ';'
    query2 = 'SELECT signal_elec3_subelec1, signal_elec3_subelec2, \
             signal_elec3_subelec3, signal_elec4_subelec1, signal_elec4_subelec2, \
             signal_elec4_subelec3 FROM ' + table + ' WHERE EEG_data_section=2 ' + user_query + ';'
    section_one = get_data(db, query1)
    #logger.info('Got section one of data - size: %s' % str(np.shape(section_one)))
    section_two = get_data(db, query2)
    #logger.info('Got section two of data - size: %s' % str(np.shape(section_two)))
    for i in range(len(section_one)):
        logger.info("Getting signals for word  = %d " %i)
        check_missing_part = len(word)
        for j in range(NUM_ELECTRODES):
            word.extend(float_arr(section_one[i][j]))
        for k in range(NUM_ELECTRODES):
            word.extend(float_arr(section_two[i][k]))
        # ignore missing words / words with missing data_section
        if len(word) == check_missing_part:
            logger.info('Missing data section - skipping current word')
            continue
        signals.append(np.asarray(word, dtype=np.float))
        word = []
    return signals


# create float array from str data  + fix missing signals
def float_arr(string):
    fix = False
    to_array = string.split(',')
    for i in range(len(to_array)):
        # ignore missing words
        if 'undefined' == to_array[i]:
            to_array = np.zeros(NUM_FEATURES,np.float)
            return to_array
        # mark the places with missing signals
        if to_array[i] in ['', '.', '-', ' ',',']:
            to_array[i] = np.nan
            fix = True
            continue
        to_array[i] = np.float(to_array[i])
    # add place holders for missing signals if array contains < NUM_FEATURES
    while len(to_array) < NUM_FEATURES:
        to_array.append(np.nan)
        fix = True
    if fix:
        to_array = fix_missing_signals(to_array)
    return to_array


# get results from DB
def get_results(db ,user_query='',table='data_set'):
    logger.info('In get results: using table - %s' % table)
    results = []
    if table!='untagged_predictions':
        if user_query:
            user_query = 'AND ' + user_query
        user_query = ' WHERE EEG_data_section=1 ' + user_query
    else :
        user_query = ' WHERE ' + user_query
    print(user_query)
    query = 'SELECT stm, stm_confidence_level, stm_remember_know, ltm, \
             ltm_confidence_level, ltm_remember_know FROM ' + table + user_query
    print(query)
    data_set = get_data(db, query)
    print(np.shape(data_set))
    for row in data_set:
        # ignore missing words
        if row[1] == 0 or row[4] == 0:
            print("no results")
            #continue
        results.append(np.array(row, int))
    return results


# fix missing signals from one electrode
def fix_missing_signals(electrode):
    imp = Imputer(axis=1, copy=False, missing_values='NaN', strategy='mean', verbose=0)
    imp.fit([electrode])
    return np.reshape(imp.transform([electrode]), NUM_FEATURES)


# functions for choosing parameters - averaged electrode , duration
def choose_signals(db, elec, duration,user_query='', table='data_set'):
    logger.info('In choose signals')
    if user_query:
        user_query = 'AND ' + user_query
    signals = []
    word = []
    average_signal =[]
    if elec in [1,2]:
        section = 1
    else:
        section = 2
    part_1 = 'SELECT signal_elec%s_subelec1 FROM ' %elec +table+ ' WHERE EEG_data_section=%s '% section + user_query +';'
    part_2 = 'SELECT signal_elec%s_subelec2 FROM ' %elec +table +' WHERE EEG_data_section=%s  '% section +user_query+';'
    part_3 = 'SELECT signal_elec%s_subelec3 FROM ' %elec +table + ' WHERE EEG_data_section=%s '% section+user_query +';'
    subelec_1 = get_data(db, part_1)
    subelec_2 = get_data(db, part_2)
    subelec_3 = get_data(db, part_3)
    for i in range(len(subelec_1)):
        logger.info('Getting signals for word -%d' % (i+1))
        average_signal.append(float_arr_length(subelec_1[i][0], duration))
        average_signal.append(float_arr_length(subelec_2[i][0], duration))
        average_signal.append(float_arr_length(subelec_3[i][0], duration))
        average_signal = np.asarray(average_signal)
        word = np.mean(average_signal,axis=0)
        logger.info('Averaged sub-electrodes for main electrode: %d , sampling %d points' % (elec, duration))
        signals.append(np.asarray(word, dtype=np.float))
        average_signal = []
    return signals


def float_arr_length(string, duration):
    fix = False
    to_array = string.split(',')
    for i in range(duration):
        # ignore missing words
        if 'undefined' == to_array[i]:
            to_array = np.zeros(duration,np.float)
            return to_array
        # mark the places with missing signals
        if to_array[i] in ['', '.', '-', ' ']:
            to_array[i] = np.nan
            fix = True
            continue
        to_array[i] = np.float(to_array[i])
    # add place holders for missing signals if array contains < NUM_SAMPLES (duration)
    while len(to_array) < duration:
        to_array.append(np.nan)
        fix = True
    if fix:
        to_array = fix_missing_signals(to_array)
    to_array = np.array(to_array[:duration],dtype=float)
    return to_array
