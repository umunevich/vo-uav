from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routers.stream import router as stream_router
from src.routers.configs import router as configs_router

app = FastAPI(
    title="Visual Odometry API",
    description="Backend for Monocular VO project",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ok", "service": "VO Backend is running"}

app.include_router(stream_router)
app.include_router(configs_router)