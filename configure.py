import os
from dotenv import load_dotenv

load_dotenv()


class minioConfigs:
    MINIO_PORT = 9000
    MINIO_HOST = "localhost"
    ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
    SECRET_KEY = os.getenv('MINIO_SECRET_KEY')

class mongoDBConfigs:
    MONGODB_HOST = "localhost"
    MONGODB_PORT = 27017
    MONGODB_NAME = "khoaHocDuLieu"

class fastAPIConfigs:
    FASTAPI_PORT = os.getenv("PORT")
    FASTAPT_HOST = "localhost"
    FASTAPI_DOMAIN = f"http://{FASTAPT_HOST}:{FASTAPI_PORT}"

class WeightConfigs:
    FACE_DETECT_WEIGHT = "D:/prj_python/backend/src/tasks/weights/face_detect.pt"
    SCRFD_WEIGHT = "D:/prj_python/backend/src/tasks/weights/scrfd_2.5g_bnkps.onnx"
    ARCFACE_R100_WEIGHT = "D:/prj_python/backend/src/tasks/weights/arcface_r100.pth"
    TRACKING_CONFIG = "D:/prj_python/backend/src/tasks/faceRecognization/configs/config_tracking.yaml"

class QdrantConfigs:
    QDRANT_HOST = "localhost"
    QDRANT_PORT = "6333"
    QDRANT_VECTOR_SIZE = 512


class LocalConfigs:
    ROOT_PATH = "D:/prj_python/backend"