r"""FastAPI AI 추론 서버의 시작점.

실행 (ai_service 폴더 안에서):
    .venv\Scripts\python -m uvicorn main:app --host 127.0.0.1 --port 8001
문서(Swagger UI): http://localhost:8001/docs
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# ai_service 폴더를 import 경로에 넣는다 → "from src..." "from services..."가 어디서 실행해도 동작
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI  # noqa: E402

from routers.predictions import router as predictions_router  # noqa: E402
from services.model_loader import get_bundle, init_bundle  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버가 켜질 때 한 번만 모델을 불러 둔다 (요청마다 다시 읽지 않음). 검사에 실패하면 서버가 뜨지 않는다.
    init_bundle()
    yield


app = FastAPI(title="Vision Action AI Service", version="0.1.0", lifespan=lifespan)
app.include_router(predictions_router)


@app.get("/health")
def health():
    """FastAPI가 살아 있는지, 추론에 필요한 모델이 준비됐는지 알려 준다."""
    bundle = get_bundle()
    if bundle is None:
        return {"status": "starting", "model_ready": False}
    return {
        "status": "ok",
        "model_ready": True,
        "classes": bundle.classes,
        "feature_count": int(bundle.classifier.n_features_in_),
        "sequence_steps": bundle.sequence_steps,
        "min_valid_frames": bundle.min_valid_frames,
        "device": str(bundle.device),
    }
