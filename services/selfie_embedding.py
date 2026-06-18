import cv2
import numpy as np

from numpy import dot
from numpy.linalg import norm

from face_model import face_app


def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def buffer_to_image(image_bytes):
    arr = np.frombuffer(image_bytes, np.uint8)

    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def generate_selfie_embedding(selfie_bytes):

    img = buffer_to_image(image_bytes=selfie_bytes)

    if img is None:
        return None

    faces = face_app.get(img)

    if len(faces) == 0:
        return None

    return faces[0].embedding.tolist()


def find_matching_photo_ids(face_embeddings, selfie_embedding):

    threshold = 0.6

    photo_ids = set()

    for face in face_embeddings:

        similarity = cosine_similarity(
            face["embedding"],
            selfie_embedding
        )
        print(similarity)
        if similarity >= threshold:
            photo_ids.add(face["photo_id"])

        elif 0.5 < similarity < threshold:
            photo_ids.add(face["photo_id"])

    return list(photo_ids)