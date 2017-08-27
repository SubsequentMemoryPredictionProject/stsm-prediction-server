import mysql.connector
import config as cfg


# get data from DB
def get_data(config_details):
    db = mysql.connector.connect(**config_details)
    cursor = db.cursor()
    query = 'SELECT * FROM data_set;'
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    db.close()
    return data


# insert data to DB
def insert_data(config_details):
    db = mysql.connector.connect(**config_details)
    cursor = db.cursor()
    query = "INSERT INTO rotem_test VALUES ('20','rotemj');"
    print("insert data")
    cursor.execute(query)
    cursor.close()
    db.commit()
    db.close()
    return


insert_data(cfg.mysql)
test = get_data(cfg.mysql)
for row in test:
    print(row)
