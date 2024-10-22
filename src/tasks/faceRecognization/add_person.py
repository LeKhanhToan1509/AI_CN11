import sys 
sys.path.append('D:/prj_python/backend/src/tasks/faceRecognization')
import argparse
import os
import shutil

import cv2
import numpy as np
import torch
from torchvision import transforms

from detector import SCRFD
from models.Iresnet import iresnet_inference
from utils import read_features
# import configure

# SCRFD_WEIGHT = configure.WeightConfigs.SCRFD_WEIGHT
# ARCFACE_WEIGHT = configure.WeightConfigs.ARCFACE_R100_WEIGHT
SCRFD_WEIGHT = "D:/prj_python/backend/src/tasks/weights/scrfd_2.5g_bnkps.onnx"
ARCFACE_WEIGHT = "D:/prj_python/backend/src/tasks/weights/arcface_r100.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detector = SCRFD(model_file=SCRFD_WEIGHT)

# Initialize the face recognizer
recognizer = iresnet_inference(
    model_name="r100", path=ARCFACE_WEIGHT, device=device
)


@torch.no_grad()
def get_feature(face_image):
    """
    Extract facial features from an image using the face recognition model.

    Args:
        face_image (numpy.ndarray): Input facial image.

    Returns:
        numpy.ndarray: Extracted facial features.
    """
    # Define a series of image preprocessing steps
    face_preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Convert the image to RGB format
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # Apply the defined preprocessing to the image
    face_image = face_preprocess(face_image).unsqueeze(0).to(device)

    # Use the model to obtain facial features
    emb_img_face = recognizer(face_image)[0].cpu().numpy()

    # Normalize the features
    images_emb = emb_img_face / np.linalg.norm(emb_img_face)
    # print(len(images_emb))
    # print(images_emb)
    return images_emb


def add_persons(backup_dir, add_persons_dir, faces_save_dir, features_path, qdrant_client):
    """
    Add a new person to the face recognition database.

    Args:
        backup_dir (str): Directory to save backup data.
        add_persons_dir (str): Directory containing images of the new person.
        faces_save_dir (str): Directory to save the extracted faces.
        features_path (str): Path to save face features.
    """
    # Initialize lists to store names and features of added images
    images_name = []
    images_emb = []

    # Read the folder with images of the new person, extract faces, and save them
    for name_person in os.listdir(add_persons_dir):
        person_image_path = os.path.join(add_persons_dir, name_person)
        print(f"name_person: {name_person}, type: {type(name_person)}")
        print(f"faces_save_dir: {faces_save_dir}, type: {type(faces_save_dir)}")
        # Create a directory to save the faces of the person
        person_face_path = os.path.join(faces_save_dir, name_person)
        os.makedirs(person_face_path, exist_ok=True)

        for image_name in os.listdir(person_image_path):
            if image_name.endswith(("png", "jpg", "jpeg")):
                input_image = cv2.imread(os.path.join(person_image_path, image_name))

                # Detect faces and landmarks using the face detector
                bboxes, landmarks = detector.detect(image=input_image)

                # Extract faces
                for i in range(len(bboxes)):
                    # Get the number of files in the person's path
                    number_files = len(os.listdir(person_face_path))

                    # Get the location of the face
                    x1, y1, x2, y2, score = bboxes[i]

                    # Extract the face from the image
                    face_image = input_image[y1:y2, x1:x2]

                    # Path to save the face
                    path_save_face = os.path.join(person_face_path, f"{number_files}.jpg")

                    # Save the face to the database
                    cv2.imwrite(path_save_face, face_image)

                    # Extract features from the face
                    images_emb.append(get_feature(face_image=face_image))
                    images_name.append(name_person)

    if images_emb == [] and images_name == []:
        print("No new person found!")
        return None

    images_emb = np.array(images_emb)
    images_name = np.array(images_name)

    features = read_features(features_path)

    if features is not None:
        old_images_name, old_images_emb = features

        images_name = np.hstack((old_images_name, images_name))
        images_emb = np.vstack((old_images_emb, images_emb))
        print("Update features!")

    # Save the combined features
    np.savez_compressed(features_path, images_name=images_name, images_emb=images_emb)

    # Move the data of the new person to the backup data directory
    for sub_dir in os.listdir(add_persons_dir):
        dir_to_move = os.path.join(add_persons_dir, sub_dir)
        shutil.move(dir_to_move, backup_dir, copy_function=shutil.copytree)
    for item in os.listdir(add_persons_dir):
        if(os.path.exists(os.path.join(add_persons_dir, item))):
            os.remove(os.path.join(add_persons_dir, item))

    print("Successfully added new person!")




def delete_person(person_name: str, backup_dir, faces_save_dir, features_path):
    images_name, images_emb = read_features(features_path)

    idx = np.where(images_name == person_name)

    if idx[0].size == 0:
        print("Person not found")
        return None

    images_name = np.delete(images_name, idx)
    images_emb = np.delete(images_emb, idx, 0)
    
    person_backup_path = os.path.join(backup_dir, person_name)
    if os.path.exists(person_backup_path):
        shutil.rmtree(person_backup_path)  
    else:
        print(f"{person_backup_path} is not found!")
    
    person_faces_path = os.path.join(faces_save_dir, person_name)
    if os.path.exists(person_faces_path):
        shutil.rmtree(person_faces_path)
    else:
        print(f"{person_faces_path} is not found!")
    
    np.savez_compressed(features_path, images_name=images_name, images_emb=images_emb)
    
    return idx
