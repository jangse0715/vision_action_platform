"""1단계 확인 스크립트 (T01 모델 불러오기)

실행 위치: ai_service 폴더 안
실행 명령: python scripts/check_model.py

하는 일
  1) 파이썬과 주요 라이브러리 버전을 출력한다 (원본 실습과 같은지 눈으로 비교)
  2) 6일차 Random Forest 모델과 JSON 두 개를 불러와 서로 맞는지 검사한다
  3) YOLO Pose 가중치를 불러올 수 있는지 검사한다
  4) 복사해 온 src 모듈을 import할 수 있는지 검사한다
모든 검사를 통과하면 마지막 줄에 "1단계 통과"가 나오고, 하나라도 실패하면 "1단계 실패"가 나온다.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

# ai_service 폴더를 기준으로 경로를 잡는다 (이 파일은 ai_service/scripts/ 안에 있음)
AI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0, str(AI_ROOT)
)  # "from src.xxx import ..."가 되도록 ai_service를 검색 경로에 넣음

MODELS = AI_ROOT / "models"
MODEL_PATH = MODELS / "rf_action_baseline.joblib"
COLUMNS_PATH = MODELS / "rf_feature_columns.json"
INFO_PATH = MODELS / "rf_training_info.json"
POSE_PATH = MODELS / "yolo11n-pose.pt"

EXPECTED_CLASSES = 6
FEATURES_PER_STEP = 38  # 관절 17개의 x·y 34개 + 팔꿈치·무릎 각도 4개

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    mark = "통과" if ok else "실패"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))


print("== 1. 버전 ==")
print("Python", sys.version.split()[0])

for lib in ("sklearn", "joblib", "numpy", "pandas", "torch", "ultralytics", "cv2"):
    try:
        mod = __import__(lib)
        print(f"{lib:12s}", getattr(mod, "__version__", "?"))
    except Exception as exc:  # 설치가 안 된 경우
        check(f"{lib} import", False, repr(exc))

print("\n== 2. 모델과 JSON ==")
for path in (MODEL_PATH, COLUMNS_PATH, INFO_PATH, POSE_PATH):
    check(f"파일 있음: models/{path.name}", path.exists())

if MODEL_PATH.exists() and COLUMNS_PATH.exists() and INFO_PATH.exists():
    import joblib

    # 버전이 달라서 나는 경고(InconsistentVersionWarning 등)를 잡아서 보여 준다
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = joblib.load(MODEL_PATH)  # 본인이 만든 신뢰할 수 있는 파일에만 사용

    version_warnings = [
        str(w.message) for w in caught if "version" in str(w.message).lower()
    ]
    check(
        "joblib 불러올 때 버전 경고 없음",
        not version_warnings,
        "; ".join(version_warnings)[:300],
    )

    columns = json.loads(COLUMNS_PATH.read_text(encoding="utf-8"))
    info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
    classes = [str(c) for c in model.classes_]
    steps = info.get("sequence_steps")

    print("모델 종류:", type(model).__name__)
    print("model.classes_:", classes)
    print("model.n_features_in_:", model.n_features_in_)
    print("sequence_steps:", steps, "| feature_count:", info.get("feature_count"))

    check("Class 6개", len(classes) == EXPECTED_CLASSES, str(len(classes)))
    check("Class 순서가 학습 정보와 같음", classes == info.get("classes"))
    check(
        "모델 입력 Feature 1,140개",
        model.n_features_in_ == 1140,
        str(model.n_features_in_),
    )
    check(
        "Feature 이름 JSON도 1,140개",
        len(columns) == model.n_features_in_,
        str(len(columns)),
    )
    check(
        "30 Step × 38 = 1,140",
        steps is not None and steps * FEATURES_PER_STEP == model.n_features_in_,
    )
    check(
        "Feature 이름이 t00~t29 순서로 놓임",
        columns[0].startswith("t00_") and columns[-1].startswith(f"t{steps - 1:02d}_"),
    )

    # 모델 안에 열 이름이 저장되어 있지 않으면, 순서를 지켜 주는 것은 JSON뿐이다
    has_names = hasattr(model, "feature_names_in_")
    print(
        "모델 안에 열 이름 저장됨(feature_names_in_):",
        has_names,
        "→ 순서는 rf_feature_columns.json으로 지켜야 함" if not has_names else "",
    )

print("\n== 3. YOLO Pose 가중치 ==")

if POSE_PATH.exists():
    try:
        from ultralytics import YOLO

        pose = YOLO(str(POSE_PATH))
        task = getattr(pose, "task", "?")
        check("YOLO Pose 불러오기", task == "pose", f"task={task}")

    except Exception as exc:
        check("YOLO Pose 불러오기", False, repr(exc)[:300])

print("\n== 4. 복사해 온 src 모듈 import ==")

for name in (
    "src.common",
    "src.pose_utils",
    "src.feature_utils",
    "src.sequence_utils",
    "src.realtime_utils",
):
    try:
        __import__(name)
        check(f"import {name}", True)
    except Exception as exc:
        check(f"import {name}", False, repr(exc)[:300])

failed = [r for r in results if not r[1]]

print(
    "\n" + ("1단계 통과 (T01)" if not failed else f"1단계 실패: {len(failed)}개 항목")
)

sys.exit(0 if not failed else 1)
