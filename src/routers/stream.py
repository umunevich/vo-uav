import logging
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
import base64
import json

# Прибираємо імпорти FeatureExtractor та FeatureMatcher!
from src.vo import VisualOdometry

logger = logging.getLogger(__name__)
router = APIRouter()

K_EUROC = np.array([
    [458.654, 0.0, 367.215],
    [0.0, 457.296, 248.375],
    [0.0, 0.0, 1.0]
])

@router.websocket("/ws/vo-stream")
async def vo_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Frontend connected. Waiting for frames...")
    
    # Ініціалізуємо НОВИЙ клас, передаючи ЛИШЕ матрицю камери
    vo = VisualOdometry(K_EUROC)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                if "," in data:
                    data = data.split(",")[1]
                    
                img_bytes = base64.b64decode(data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

                if frame is None:
                    logger.warning("Received invalid frame data. Skipping.")
                    continue

                current_pose = await asyncio.to_thread(vo.process_frame, frame)

                response = {
                    "x": float(current_pose[0][0]),
                    "y": float(current_pose[1][0]),
                    "z": float(current_pose[2][0])
                }

                await websocket.send_text(json.dumps(response))

            except Exception as frame_err:
                logger.error(f"Error processing single frame: {frame_err}")

    except WebSocketDisconnect:
        logger.info("Client disconnected. Session ended. Cleaning up memory...")
    except Exception as e:
        logger.error(f"Unexpected connection error: {e}")