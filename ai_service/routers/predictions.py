"""영상 분석 API.

POST /api/v1/predict/  — Celery Worker(나중 단계)나 Swagger UI가 영상 파일을 보내면
                          사람별 동작 예측을 JSON으로 돌려준다. DB에는 아무것도 저장하지 않는다.

Worker가 결과를 구분하는 규칙 (과제 5절 '구현 전에 정할 문제' 2번에 대한 답)
  - HTTP 422 : 입력 자체가 문제 (영상이 아님, 빈 파일, 열 수 없는 영상) → 작업 FAILED
  - HTTP 200 + status="ok"         : 한 사람 이상 예측함 → 작업 COMPLETED, 예측 저장
                                     (primary_track_id = 박스 신뢰도 합이 가장 큰 사람, 화면은 이 사람 중심)
  - HTTP 200 + status="no_person"  : 영상은 정상인데 사람이 없음 → 작업 COMPLETED, 예측 0건
  - HTTP 200 + status="too_short"  : 사람은 보였지만 유효 프레임이 10개 미만 → 작업 COMPLETED, 예측 0건
  - HTTP 503 : 모델이 아직 준비되지 않음 → 잠시 뒤 다시 시도할 일
"""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from services.action_inference import predict_actions
from services.model_loader import get_bundle
from services.pose_pipeline import VideoReadError, extract_person_sequences

router = APIRouter(prefix="/api/v1", tags=["predictions"])

ALLOWED_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


@router.post("/predict/")
def predict_video(
    video: UploadFile = File(..., description="분석할 영상 파일 (mp4 등)"),
    mode: str = Query("track", pattern="^(track|single)$",
                      description="track: 사람마다 Track ID로 분리 / single: 학습 때처럼 신뢰도 1등 한 사람만"),
):
    # async가 아닌 일반 함수라서 FastAPI가 별도 스레드에서 실행한다 (영상 분석이 서버 전체를 붙잡지 않게)
    bundle = get_bundle()
    if bundle is None:
        raise _error(503, "MODEL_NOT_READY", "모델이 아직 준비되지 않았습니다.")

    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise _error(422, "UNSUPPORTED_FILE", f"영상 파일만 받습니다 ({', '.join(sorted(ALLOWED_SUFFIXES))}).")

    started = time.perf_counter()
    # 업로드된 파일을 임시 파일로 저장한다 (OpenCV는 파일 경로로 영상을 읽기 때문)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        if tmp_path.stat().st_size == 0:
            raise _error(422, "EMPTY_FILE", "빈 파일입니다.")
        try:
            analysis = extract_person_sequences(tmp_path, bundle, mode=mode)
        except VideoReadError as exc:
            raise _error(422, "VIDEO_UNREADABLE", str(exc))
        predictions = predict_actions(analysis, bundle)
    finally:
        tmp_path.unlink(missing_ok=True)

    primary = next((p for p in predictions if p["is_primary"]), None)
    if predictions:
        status = "ok"
        message = f"{len(predictions)}명의 동작을 예측했습니다. 주인공(Track {primary['track_id']}): {primary['action_label']}"
    elif analysis.skipped:
        status = "too_short"
        message = (f"사람은 보였지만 동작을 판단할 만큼 유효한 프레임이 부족합니다 "
                   f"(사람마다 {bundle.min_valid_frames}프레임 이상 필요).")
    else:
        status, message = "no_person", "영상에서 사람을 찾지 못했습니다."

    return {
        "status": status,
        "message": message,
        "mode": mode,
        "video": {
            "filename": video.filename,
            "frame_count": analysis.frame_count,
            "fps": round(analysis.fps, 3),
            "width": analysis.width,
            "height": analysis.height,
        },
        "primary_track_id": primary["track_id"] if primary else None,
        "predictions": predictions,
        "skipped_tracks": [s.__dict__ for s in analysis.skipped],
        "processing_ms": int((time.perf_counter() - started) * 1000),
    }
