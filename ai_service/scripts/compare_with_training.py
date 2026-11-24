r"""2단계 검사: 서비스가 만든 숫자 1,140개가 학습 때 숫자와 같은지 맞대 본다.

실행 (ai_service 폴더 안에서):
    .venv\Scripts\python scripts\compare_with_training.py <S01 영상 폴더> <train.csv 경로>
예:
    .venv\Scripts\python scripts\compare_with_training.py ^
        "D:\obsidian\...\원본_복사\data\S01" ^
        "D:\obsidian\...\원본_복사\data\day06\sequences\train.csv"

영상마다 두 방식으로 숫자를 만든다.
  single : 학습 때(26번)처럼 신뢰도 1등 한 사람만 → 학습 숫자와 거의 같아야 정상
  track  : 서비스 기본값(ByteTrack) → single과 같거나 아주 조금 다를 수 있다
그리고 학습 숫자와의 차이(최대·평균 절댓값)와 예측 결과를 출력한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from services.action_inference import predict_actions  # noqa: E402
from services.model_loader import load_bundle  # noqa: E402
from services.pose_pipeline import extract_person_sequences  # noqa: E402

CLOSE_ENOUGH = 1e-3   # 이 값보다 작으면 "같다"고 본다 (반올림·부동소수점 차이 정도)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    video_dir, train_csv = Path(sys.argv[1]), Path(sys.argv[2])
    bundle = load_bundle()
    train = pd.read_csv(train_csv)
    columns = bundle.feature_columns          # 1,140개 열 이름 = 모델이 기대하는 순서

    all_ok = True
    for video in sorted(video_dir.glob("*.mp4")):
        clip_id = video.stem
        hit = train[train["clip_id"] == clip_id]
        if hit.empty:
            print(f"- {clip_id}: train.csv에 없음 (학습에 안 쓰인 영상) → 예측만 본다")
            reference = None
            label = None
        else:
            reference = hit.iloc[0][columns].to_numpy(dtype=np.float32)
            label = hit.iloc[0]["action_label"]
        print(f"\n== {clip_id}  (정답: {label})")
        for mode in ("single", "track"):
            analysis = extract_person_sequences(video, bundle, mode=mode)
            preds = predict_actions(analysis, bundle)
            if not analysis.persons:
                print(f"  [{mode}] 사람 없음 / 건너뜀: {[s.__dict__ for s in analysis.skipped]}")
                all_ok = False
                continue
            person = max(analysis.persons, key=lambda p: p.valid_frames)   # 가장 오래 보인 사람
            pred = next(p for p in preds if p["track_id"] == person.track_id)
            line = (f"  [{mode}] Track {person.track_id}: 예측 {pred['action_label']} "
                    f"({pred['confidence']:.3f}), 유효 프레임 {person.valid_frames}")
            if len(analysis.persons) > 1:
                line += f", 사람 {len(analysis.persons)}명 감지"
            if reference is not None:
                diff = np.abs(person.vector - reference)
                same = float(diff.max()) < CLOSE_ENOUGH
                line += f" | 학습 숫자와 차이: 최대 {diff.max():.6f}, 평균 {diff.mean():.6f} → {'같음' if same else '다름'}"
                correct = pred["action_label"] == label
                line += f" | 예측 {'맞음' if correct else '틀림'}"
                if mode == "single" and not (same and correct):
                    all_ok = False
            print(line)

    print("\n" + ("통과: single 방식이 학습 숫자와 같고 예측도 맞음" if all_ok else "확인 필요: 위에서 '다름' 또는 '틀림'인 줄을 보자"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
