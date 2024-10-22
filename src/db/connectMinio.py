from configure import minioConfigs
from minio import Minio

class ConnectMinio:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConnectMinio, cls).__new__(cls)
            cls._instance.endpoint = f"{minioConfigs.MINIO_HOST:}:{minioConfigs.MINIO_PORT}"
            cls._instance.access_key = minioConfigs.ACCESS_KEY
            cls._instance.secret_key = minioConfigs.SECRET_KEY
            cls._instance.connection = None

        return cls._instance

    def connect(self):
        if self.connection is None:
            self.connection = Minio(self.endpoint, access_key=self.access_key, secret_key=self.secret_key, secure=False)
            if(self.connection):
                print("Connected to Minio successfully")
        return self.connection
