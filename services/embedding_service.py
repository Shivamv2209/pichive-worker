import cv2
import numpy as np
import requests

from face_model import face_app


def url_to_image(url):
    session = requests.Session()
    response = session.get(url)
    response.raise_for_status()

    arr = np.frombuffer(response.content, np.uint8)

    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def generate_embeddings(photos):
    face_embeddings = []

    

    for photo in photos:
        
        photo_id = photo["photo_id"]
        url = photo["url"]

        
        img = url_to_image(url)
        

        if img is None:
            continue

            
        faces = face_app.get(img)
        

        for i, face in enumerate(faces):
            face_embeddings.append({
                "photo_id": photo_id,
                "face_index": i,
                "embedding": face.embedding.tolist()
            })
        
            

        

    return face_embeddings