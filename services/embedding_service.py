import cv2
import numpy as np
import requests
import time

from face_model import face_app


def url_to_image(url):
    response = requests.get(url)
    response.raise_for_status()

    arr = np.frombuffer(response.content, np.uint8)

    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def generate_embeddings(photos):
    face_embeddings = []

    overall_start = time.time()

    for photo in photos:
        photo_start = time.time()
        photo_id = photo["photo_id"]
        url = photo["url"]

        download_start = time.time()
        img = url_to_image(url)
        print(f"Download + Decode: {time.time() - download_start:.2f}s")

        if img is None:
            continue

        embedding_start = time.time()    
        faces = face_app.get(img)
        print(f"Face Detection + Embedding: {time.time() - embedding_start:.2f}s")

        for i, face in enumerate(faces):
            face_embeddings.append({
                "photo_id": photo_id,
                "face_index": i,
                "embedding": face.embedding.tolist()
            })
        print(f"Total for photo: {time.time() - photo_start:.2f}s")
        print("-" * 40)    

    print(f"TOTAL TIME: {time.time() - overall_start:.2f}s")    

    return face_embeddings