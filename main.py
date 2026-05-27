from fastapi import FastAPI
from src.routers.stream import router as stream_router

app = FastAPI(
    title="Visual Odometry API",
    description="Backend for Monocular VO project"
)

app.include_router(stream_router)

@app.get("/")
def health_check():
    return {"status": "ok", "service": "VO Backend is running"}