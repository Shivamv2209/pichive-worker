from insightface.app import FaceAnalysis

face_app = FaceAnalysis(name="buffalo_s",allowed_modules=["detection", "recognition"])
face_app.prepare(ctx_id=-1)