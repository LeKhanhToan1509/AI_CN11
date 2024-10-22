# APIs minio server
from db.connectMinio import ConnectMinio
from db.connectMongo import ConnectMongoDb
from helpers.checkBucket import Bucket
import json, os, cv2
import time

class MinioServer:
    def __init__(self):
        self.mongoClient = ConnectMongoDb().connect()


    @staticmethod
    def create_bucket(bucket_name: str) -> int:
        bucket_check = Bucket()
        if not bucket_check.check_name(bucket_name):
            print("Bucket name is invalid")
            return 0
        
        minio_client = ConnectMinio().connect()
        if bucket_check.check_exists(bucket_name):
            print(f"Bucket '{bucket_name}' already exists")
            return 0
        
        minio_client.make_bucket(bucket_name)
        print(f"Bucket '{bucket_name}' created successfully")

    @staticmethod
    def upload_file(file_path: str, destination_file: str, bucket_name: str = 'prj_python') -> int:
        try:
            if not os.path.exists(file_path):
                print(f"File '{file_path}' not found")
                return 0
            if not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                print(f"File '{file_path}' is not an image")
                return
            
            img = cv2.imread(file_path)
            img_width, img_height = img.shape[1], img.shape[0]

            client = ConnectMinio().connect()
            policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"AWS": "*"},
                        "Action": ["s3:GetObject"],
                        "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
                    }
                ]
            }
            client.set_bucket_policy(bucket_name, json.dumps(policy))
            url = client.presigned_get_object(bucket_name, destination_file).split('?')[0]
            client.fput_object(
                bucket_name, 
                destination_file, 
                file_path 
            )
            print(
                f"File '{file_path}' successfully uploaded as '{destination_file}' to bucket '{bucket_name}'"
            )
        except Exception as e:
            print(f"Failed to upload file '{file_path}' to bucket '{bucket_name}': {e}")
            return 0
        finally:
            return {
                'link': url,
                'img_width': img_width,
                'image_height': img_height,
                'date': time.now()
            }
