from fastapi import FastAPI
from typing import List,Dict,Any

from services.embedding_service import generate_embeddings
from services.selfie_embedding import (
    generate_selfie_embedding,
    find_matching_photo_ids
)

app = FastAPI()


@app.get("/")
def root():
    return {
        "message": "Worker running"
    }


@app.post("/generate-embeddings")
async def embeddings(data: List[Dict[str,Any]]):


    print("received",data)
    embeddings = generate_embeddings(data)

    return {
        "embeddings": embeddings
    }


@app.post("/search-selfie")
async def search(data: dict):

    selfie_bytes = bytes(data["selfie"])

    face_embeddings = data["face_embeddings"]

    selfie_embedding = generate_selfie_embedding(selfie_bytes)

    if selfie_embedding is None:
        return {
            "photo_ids": []
        }

    photo_ids = find_matching_photo_ids(
        face_embeddings,
        selfie_embedding
    )

    return {
        "photo_ids": photo_ids
    }