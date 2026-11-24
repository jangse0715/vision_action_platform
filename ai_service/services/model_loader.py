"""학습 산출물(모델·JSON)과 YOLO Pose 가중치를 서버 시작 때 한 번만 불러 두는 곳.

- 1단계 확인 스크립트(scripts/check_model.py)가 검사한 내용을 서버 시작 때 다시 검사한다.
  검사에 실패하면 서버가 아예 뜨지 않게 해서, 틀린 모델로 추론하는 일을 막는다.
- 불러 둔 것은 ModelBundle 하나에 모아 두고, 요청마다 다시 읽지 않고 재사용한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
from ultralytics import YOLO

from src.common import load_config, resolve_device

AI_ROOT = Path(__file__).resolve().parents[1]  # ai_service 폴더
MODELS_DIR = AI_ROOT / "models"
CONFIGS_DIR = AI_ROOT / "configs"

FEATURES_PER_STEP = 38  # 관절 17개의 x·y 34개 + 팔꿈치·무릎 각도 4개


@dataclass
class ModelBundle:
    classifier: object  # 6일차 Random Forest
    feature_columns: list[str]  # 1,140개 Feature의 이름과 순서
    classes: list[str]  # model.classes_ (확률 순서의 이름표)
    sequence_steps: int  # 30
    min_valid_frames: int  # 10 (학습 때 이보다 적으면 뺐다)
    pose_weights: Path  # YOLO Pose 가중치 파일 경로
    pose_model: YOLO  # 서버 시작 때 불러 둔 YOLO (사람 1명 모드에서 사용)
    project_config: dict  # configs/project.json (이미지 크기, 신뢰도 등)
    realtime_config: dict  # configs/day07_realtime.json (판단 불확실 기준 등)
    tracking_config: dict  # configs/day08_multi_person.json (추적기 이름)
    device: object  # "cpu" 또는 GPU 번호
    info: dict = field(default_factory=dict)


_bundle: ModelBundle | None = None


def _read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def load_bundle() -> ModelBundle:
    """모델과 설정을 불러와 검사한 뒤 ModelBundle로 돌려준다. 문제가 있으면 RuntimeError."""
    project_config = (
        load_config()
    )  # src/common.py가 configs/project.json을 읽는다 (원본 그대로)
    training_info = _read_json(MODELS_DIR / "rf_training_info.json")
    feature_columns = _read_json(MODELS_DIR / "rf_feature_columns.json")

    # 본인이 만든 신뢰할 수 있는 파일에만 joblib.load를 쓴다 (과제 1-2)
    classifier = joblib.load(MODELS_DIR / "rf_action_baseline.joblib")

    classes = [str(c) for c in classifier.classes_]
    steps = int(training_info["sequence_steps"])
    expected = steps * FEATURES_PER_STEP

    # --- 1단계와 같은 검사 (하나라도 어긋나면 서버를 띄우지 않는다) ---
    problems = []
    if len(classes) != 6:
        problems.append(f"Class 수가 6이 아님: {len(classes)}")
    if classes != training_info.get("classes"):
        problems.append("model.classes_와 학습 정보의 Class 순서가 다름")
    if classifier.n_features_in_ != expected:
        problems.append(
            f"모델 입력 수 {classifier.n_features_in_} != {steps} × {FEATURES_PER_STEP}"
        )
    if len(feature_columns) != classifier.n_features_in_:
        problems.append(
            f"Feature 이름 수 {len(feature_columns)} != 모델 입력 수 {classifier.n_features_in_}"
        )
    if problems:
        raise RuntimeError("모델 검사 실패: " + " / ".join(problems))

    pose_weights = MODELS_DIR / project_config["pose_model"]  # yolo11n-pose.pt
    if not pose_weights.exists():
        raise RuntimeError(f"YOLO Pose 가중치가 없음: {pose_weights}")
    pose_model = YOLO(str(pose_weights))

    return ModelBundle(
        classifier=classifier,
        feature_columns=feature_columns,
        classes=classes,
        sequence_steps=steps,
        min_valid_frames=int(training_info.get("min_valid_frames", 10)),
        pose_weights=pose_weights,
        pose_model=pose_model,
        project_config=project_config,
        realtime_config=_read_json(CONFIGS_DIR / "day07_realtime.json"),
        tracking_config=_read_json(CONFIGS_DIR / "day08_multi_person.json"),
        device=resolve_device(project_config["device"]),
        info=training_info,
    )


def init_bundle() -> ModelBundle:
    """서버 시작 때 main.py가 한 번 부른다."""
    global _bundle
    _bundle = load_bundle()
    return _bundle


def get_bundle() -> ModelBundle | None:
    """요청을 처리할 때 불러 둔 ModelBundle을 꺼낸다. 아직 준비되지 않았으면 None."""
    return _bundle
