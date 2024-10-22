from pymongo import MongoClient
from configure import mongoDBConfigs

class ConnectMongoDb:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConnectMongoDb, cls).__new__(cls)
            cls._instance.host = mongoDBConfigs.MONGODB_HOST
            cls._instance.port = mongoDBConfigs.MONGODB_PORT
            cls._instance.connection = None
        return cls._instance

    def connect(self):
        if self.connection is None:
            self.connection = MongoClient(f'mongodb://{self.host}:{self.port}')
            if(self.connection):
                print("Connected to MongoDB successfully")
        return self.connection
