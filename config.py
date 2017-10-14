
mysql = {
    'host': "stsmdb.czcokqfhjscu.us-east-1.rds.amazonaws.com",
    'user': "master",
    'password': "12345678",
    'port': 3306,
    'database': "STSMDB"
  }

model_features = {
    'elec': 1,
    'duration': 256
    }

mlp_params = {
    'layers': (100, 20),
    'activation': 'identity',
    'learning_rate': 'invscaling'
}

app = {
    'port': 3100,
    'host': '0.0.0.0'
}
