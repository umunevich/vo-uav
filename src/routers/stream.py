import logging
import asyncio
import base64
import json

import cv2
import numpy as np
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from src.config_store import DEFAULT_CONFIG_ID, load_camera_matrix
from src.vo import VisualOdometry

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/vo-stream")
async def vo_stream_endpoint(
    websocket: WebSocket,
    config_id: str = Query(default=DEFAULT_CONFIG_ID, description="Camera profile ID"),
):
    await websocket.accept()

    try:
        k_matrix = load_camera_matrix(config_id)
    except Exception as exc:
        logger.warning("Failed to load config '%s', using default: %s", config_id, exc)
        k_matrix = load_camera_matrix(DEFAULT_CONFIG_ID)

    logger.info("VO stream started with camera profile '%s'", config_id)
    vo = VisualOdometry(k_matrix)

    try:
        while True:
            data = await websocket.receive_text()

            try:
                if "," in data:
                    data = data.split(",", 1)[1]

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
                    "z": float(current_pose[2][0]),
                }

                await websocket.send_text(json.dumps(response))

            except Exception as frame_err:
                logger.error("Error processing single frame: %s", frame_err)

    except WebSocketDisconnect:
        logger.info("Client disconnected. Session ended.")
    except Exception as exc:
        logger.error("Unexpected connection error: %s", exc)
