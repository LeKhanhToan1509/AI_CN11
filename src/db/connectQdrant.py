from qdrant_client import QdrantClient
from configure import QdrantConfigs


class ConnectQdrant:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConnectQdrant, cls).__new__(cls)
            cls._instance.host = QdrantConfigs.QDRANT_HOST
            cls._instance.port = QdrantConfigs.QDRANT_PORT
            cls._instance.connection = None
        return cls._instance

    def connect(self):
        if self.connection is None:
            self.connection = QdrantClient(host=self.host, port=self.port)
            if(self.connection):
                print("Connected to MongoDB successfully")
        return self.connection
