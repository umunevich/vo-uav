from fastapi import APIRouter, HTTPException, status
from fastapi.responses import PlainTextResponse

from src.schemas.trajectory import (
    EuRoCSequenceInfo,
    TrajectoryEvaluateRequest,
    TrajectoryEvaluateResponse,
    TrajectoryExportRequest,
    TrajectoryExportResponse,
)
from src.trajectory_eval import (
    evaluate_against_euroc,
    export_tum,
    list_euroc_sequences,
)

router = APIRouter(prefix="/api/trajectory", tags=["Trajectory"])


@router.get("/euroc-sequences", response_model=list[EuRoCSequenceInfo])
async def get_euroc_sequences() -> list[EuRoCSequenceInfo]:
    return [EuRoCSequenceInfo(**item) for item in list_euroc_sequences()]


@router.post("/export-tum", response_model=TrajectoryExportResponse)
async def export_tum_trajectory(body: TrajectoryExportRequest) -> TrajectoryExportResponse:
    if not body.samples:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Trajectory is empty",
        )
    try:
        result = export_tum(
            [s.model_dump() for s in body.samples],
            sequence_id=body.sequence_id,
            fps=body.fps,
            apply_global_scale=body.apply_global_scale,
        )
    except (KeyError, FileNotFoundError, ValueError, IndexError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return TrajectoryExportResponse(**result)


@router.post("/export-tum/download")
async def download_tum_trajectory(body: TrajectoryExportRequest) -> PlainTextResponse:
    response = await export_tum_trajectory(body)
    return PlainTextResponse(
        content=response.content,
        media_type="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename="{response.filename}"',
        },
    )


@router.post("/evaluate", response_model=TrajectoryEvaluateResponse)
async def evaluate_trajectory(body: TrajectoryEvaluateRequest) -> TrajectoryEvaluateResponse:
    if len(body.samples) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 3 trajectory samples are required",
        )
    try:
        metrics = evaluate_against_euroc(
            [s.model_dump() for s in body.samples],
            body.sequence_id,
        )
    except (KeyError, FileNotFoundError, ValueError, IndexError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return TrajectoryEvaluateResponse(
        sequence_id=metrics["sequence_id"],
        sequence_label=metrics["sequence_label"],
        n_samples=metrics["n_samples"],
        global_scale=metrics["global_scale"],
        ate_rmse_m=metrics["ate_rmse_m"],
        ate_mean_m=metrics["ate_mean_m"],
        ate_median_m=metrics["ate_median_m"],
        ate_max_m=metrics["ate_max_m"],
        gt_path_length_m=metrics["gt_path_length_m"],
        est_path_length_raw=metrics["est_path_length_raw"],
        est_path_length_scaled_m=metrics["est_path_length_scaled_m"],
        end_error_m=metrics["end_error_m"],
        drift_pct=metrics["drift_pct"],
        tum_raw=metrics["tum_raw"],
        tum_scaled=metrics["tum_scaled"],
        scaled_positions=metrics["scaled_positions"],
    )
