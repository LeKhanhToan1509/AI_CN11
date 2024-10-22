import sys
sys.path.append('D:/prj_python/backend')
import configure
from fastapi import FastAPI
import uvicorn
from fastapi import FastAPI, File, UploadFile
import os, shutil
import configure
from fastapi.middleware.cors import CORSMiddleware
from tasks.faceRecognization.add_person import add_persons, delete_person
from tasks.faceRecognization.utils import compare_encodings
import uuid
from qdrant_client.models import VectorParams, Distance
from db.connectQdrant import ConnectQdrant
from mimetypes import guess_extension, guess_type


DOMAIN = configure.fastAPIConfigs.FASTAPI_DOMAIN
QDRANT_VECTOR_SIZE = configure.QdrantConfigs.QDRANT_VECTOR_SIZE
PERSON_COLLECTION = "person_collection"

#global variables
add_person_path = "D:/prj_python/backend//src/tasks/faceRecognization/datasets/new_persons"
backup_dir = "D:/prj_python/backend/src/tasks/faceRecognization/datasets/backup"
add_persons_dir = "D:/prj_python/backend/src/tasks/faceRecognization/datasets/new_persons"
faces_save_dir = "D:/prj_python/backend/src/tasks/faceRecognization/datasets/data"
features_path = "D:/prj_python/backend/src/tasks/faceRecognization/datasets/face_features/feature"



qdrant_client = ConnectQdrant().connect()
if not qdrant_client.collection_exists(PERSON_COLLECTION):
   qdrant_client.create_collection(
      collection_name=PERSON_COLLECTION,
      vectors_config=VectorParams(size=QDRANT_VECTOR_SIZE, distance=Distance.COSINE),
   )

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)

@app.get('/')
def root():
    return {'message': 'Hello World'}   

@app.post("/add_persons/")
async def create_upload_file(file: UploadFile = File(...), person_name: str = None):
    if not os.path.exists(add_person_path):
        os.makedirs(add_person_path)
    if not os.path.exists(f"{add_person_path}/{person_name}"):
        os.makedirs(f"{add_person_path}/{person_name}")
    
    file_ext = '.' + file.filename.split(".")[-1].lower()
    
    # Additional image validation using mimetypes
    if not guess_type(file.filename)[0].startswith("image"):
        return {
            "message": "File not uploaded",
            "error": "File is not an image"
        }

    file_name = uuid.uuid4().hex + file_ext
    file_location = f"{add_person_path}/{person_name}/{file_name}"

    os.makedirs(os.path.dirname(file_location), exist_ok=True)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Add exception handling for add_persons
    try:
        add_persons(backup_dir, add_persons_dir, faces_save_dir, features_path, qdrant_client)
    except Exception as e:
        return {
            "message": "Failed to add person",
            "error": str(e)
        }

    return {
        "location": file_location,
        "message": "File uploaded successfully"
    }

@app.delete("/delete_persons/")
async def delete_persons(person_name: str):
    idx = delete_person(person_name, backup_dir, faces_save_dir, features_path)
    if idx is None:
        return {
            "message": f"Person {person_name} not found"
        }
    return {
        "message": f"Person {person_name} deleted successfully",
        "index": idx[0].tolist() 
    }

    

@app.post('/upload_video')
async def upload_video():
    return {
        'message': 'Uploaded video uploaded successfully'
    }


if __name__ == "__main__":
    uvicorn.run(app, host=configure.fastAPIConfigs.FASTAPT_HOST, port=int(configure.fastAPIConfigs.FASTAPI_PORT))

    
