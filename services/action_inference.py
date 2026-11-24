"""사람마다 만든 1,140개 숫자를 Random Forest에 넣어 동작과 신뢰도(Confidence)를 정리한다."""
from __future__ import annotations

import numpy as np

from services.model_loader import ModelBundle
from services.pose_pipeline import VideoAnalysis


def predict_actions(analysis: VideoAnalysis, bundle: ModelBundle) -> list[dict]:
    """사람(Track ID)마다 예측 한 건씩 만든다. 사람이 없으면 빈 목록."""
    if not analysis.persons:
        return []

    # 모델은 열 이름을 모른다(feature_names_in_ 없음). 순서는 pose_pipeline이
    # 학습 때와 같은 함수로 만들었으므로 rf_feature_columns.json의 순서와 같다.
    matrix = np.stack([person.vector for person in analysis.persons]).astype(np.float32)
    probabilities = bundle.classifier.predict_proba(matrix)

    # 7일차에 쓴 기준: 이 값보다 낮으면 화면에 '판단 불확실'로 표시하게 한다
    threshold = float(bundle.realtime_config.get("confidence_threshold", 0.55))

    predictions = []
    for person, probs in zip(analysis.persons, probabilities):
        best = int(np.argmax(probs))
        confidence = float(probs[best])
        predictions.append({
            "track_id": person.track_id,
            "is_primary": person.is_primary,            # 동작 판별의 대상(주인공)인가
            "merged_track_ids": person.merged_track_ids,  # 다시 잇기로 묶인 원래 Track 번호들
            "action_label": bundle.classes[best],
            "confidence": round(confidence, 4),
            "is_uncertain": confidence < threshold,
            "valid_frames": person.valid_frames,
            "first_frame": person.first_frame,
            "last_frame": person.last_frame,
            "window_count": 1,  # 업로드 영상은 학습 때처럼 영상 전체를 한 덩어리(Window 1개)로 본다
            "probabilities": {label: round(float(p), 4) for label, p in zip(bundle.classes, probs)},
        })
    predictions.sort(key=lambda p: (not p["is_primary"], p["track_id"]))  # 주인공을 맨 앞에
    return predictions
