import os
import re
from typing import List

import yaml
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from src.camera_calibration import calibrate_from_chessboard_images
from src.config_store import STORAGE_DIR, ensure_default_profile, get_file_path
from src.schemas.vo_config import (
    CalibrationResponse,
    VOConfigSchema,
    VOConfigShortResponse,
)

router = APIRouter(prefix="/api/configs", tags=["VO Configurations"])

ensure_default_profile()


@router.get("", response_model=List[VOConfigShortResponse])
async def list_configurations():
    configs: list[dict[str, str]] = []

    for filename in os.listdir(STORAGE_DIR):
        if not filename.endswith(".yaml"):
            continue

        config_id = filename[:-5]
        file_path = os.path.join(STORAGE_DIR, filename)

        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
                configs.append({"id": config_id, "name": data.get("name", config_id)})
        except OSError:
            continue

    configs.sort(key=lambda item: item["name"].lower())
    return configs


@router.get("/{config_id}", response_model=dict)
async def get_configuration(config_id: str):
    file_path = get_file_path(config_id)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Configuration profile not found")

    with open(file_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@router.post("/calibrate", response_model=CalibrationResponse)
async def calibrate_camera(
    inner_corners_cols: int = Form(..., ge=3, description="Inner chessboard corners along width"),
    inner_corners_rows: int = Form(..., ge=3, description="Inner chessboard corners along height"),
    square_size_mm: float = Form(..., gt=0, description="Chessboard square size in millimeters"),
    images: List[UploadFile] = File(..., description="Calibration images (chessboard at varied poses)"),
):
    if not images:
        raise HTTPException(status_code=400, detail="Upload at least one calibration image.")

    image_bytes: list[bytes] = []
    for upload in images:
        payload = await upload.read()
        if payload:
            image_bytes.append(payload)

    if not image_bytes:
        raise HTTPException(status_code=400, detail="All uploaded files were empty.")

    try:
        result = calibrate_from_chessboard_images(
            image_bytes,
            inner_corners_cols=inner_corners_cols,
            inner_corners_rows=inner_corners_rows,
            square_size_m=square_size_mm / 1000.0,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return CalibrationResponse(
        camera={
            "fu": result.fu,
            "fv": result.fv,
            "cu": result.cu,
            "cv": result.cv,
        },
        distortion=result.distortion,
        reprojection_error=result.reprojection_error,
        images_used=result.images_used,
        image_width=result.image_width,
        image_height=result.image_height,
    )


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_configuration(payload: VOConfigSchema):
    config_id = re.sub(r"[^a-zA-Z0-9]", "_", payload.name.lower()).strip("_")
    if not config_id:
        raise HTTPException(status_code=400, detail="Profile name must contain letters or digits.")

    file_path = get_file_path(config_id)

    if os.path.exists(file_path):
        raise HTTPException(
            status_code=400,
            detail=f"Profile with name or ID '{config_id}' already exists",
        )

    config_data = payload.model_dump(exclude_none=True)

    with open(file_path, "w", encoding="utf-8") as handle:
        yaml.dump(config_data, handle, default_flow_style=False, allow_unicode=True)

    return {"status": "success", "id": config_id, "message": "Profile created successfully"}


@router.put("/{config_id}")
async def update_configuration(config_id: str, payload: VOConfigSchema):
    file_path = get_file_path(config_id)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Configuration profile not found")

    config_data = payload.model_dump(exclude_none=True)

    with open(file_path, "w", encoding="utf-8") as handle:
        yaml.dump(config_data, handle, default_flow_style=False, allow_unicode=True)

    return {"status": "success", "message": f"Profile '{config_id}' updated successfully"}


@router.delete("/{config_id}")
async def delete_configuration(config_id: str):
    from src.config_store import DEFAULT_CONFIG_ID

    if config_id == DEFAULT_CONFIG_ID:
        raise HTTPException(status_code=400, detail="Cannot delete the default system profile")

    file_path = get_file_path(config_id)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Configuration profile not found")

    os.remove(file_path)
    return {"status": "success", "message": f"Profile '{config_id}' deleted successfully"}
