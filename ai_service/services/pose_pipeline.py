"""영상 한 편 → (사람마다) 숫자 1,140개.

학습 때와 같은 숫자를 만드는 것이 이 파일의 유일한 목표다.
그래서 계산은 새로 짜지 않고, 학습 때 쓴 원본 함수를 그대로 부른다.

  학습 때 (원본)                                   여기서
  26번: 프레임마다 YOLO Pose → 관절 좌표 한 줄  →  _row_from_keypoints()  (같은 열 이름, 같은 반올림)
  34번: build_frame_feature_matrix → 38개       →  같은 함수 호출
        resample_sequence → 30 Step (영상 전체)  →  같은 함수 호출
        flatten_sequence → 1,140개               →  같은 함수 호출

사람을 고르는 방식은 두 가지다.
  - mode="track"  : ByteTrack으로 여러 사람을 추적하고, 쪼개진 Track을 사람 단위로 다시 잇는다 (서비스 기본값, T04)
                    박스 신뢰도 합이 가장 큰 사람을 '주인공'(is_primary)으로 표시한다
  - mode="single" : 학습 때(26번)처럼 프레임마다 박스 신뢰도 1등 한 사람만 쓴다 (학습 때와 똑같은지 검사용)

다시 잇기 규칙 (2026-11-24, A002·SBU 영상 3편에서 실제로 재서 정함)
  ① 중복: 두 Track이 잠깐만 함께 보이고(짧은 쪽 길이의 20% 이하) 그동안 박스가 크게 겹치면(IoU 중앙값 0.4 이상)
         한 사람이 두 번 잡힌 것으로 보고 묶는다.
         - A002(다리 들기): 함께 3프레임 / 64프레임, IoU 중앙값 0.50 → 묶음
         - SBU 포옹: 함께 24 / 24프레임, IoU 중앙값 0.53 → 두 사람이 붙어 있는 것이므로 묶지 않음
           (IoU만 보면 포옹하는 두 사람을 한 사람으로 묶게 된다. '얼마나 오래 함께 보였나'가 둘을 가른다)
  ② 끊김: 한 Track이 끝나고 lost_timeout(8일차 설정 2초) 안에, 끝난 자리와 겹치는 곳(IoU 0.3 이상)에서
         새 Track이 시작되면 이어진 것으로 보고 묶는다.
  묶음 안에서는 프레임마다 박스 신뢰도 1등 박스 하나만 쓴다 (학습 때 26번과 같은 규칙).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.pose_utils import COCO_KEYPOINT_NAMES
from src.sequence_utils import build_frame_feature_matrix, flatten_sequence, resample_sequence

from services.model_loader import FEATURES_PER_STEP, ModelBundle


DUPLICATE_MAX_OVERLAP_RATIO = 0.2   # ① 함께 보인 프레임 수 / 짧은 Track 길이
DUPLICATE_MIN_IOU = 0.4             # ① 함께 보인 동안 IoU 중앙값
RECONNECT_MIN_IOU = 0.3             # ② 끝난 박스와 새로 시작한 박스의 IoU


class VideoReadError(Exception):
    """영상 파일을 열 수 없거나 프레임을 하나도 읽지 못했을 때."""


@dataclass
class PersonSequence:
    track_id: int
    vector: np.ndarray          # 모양 (1140,) — 모델에 넣을 숫자
    valid_frames: int           # 38개를 만들 수 있었던 프레임 수
    first_frame: int
    last_frame: int
    merged_track_ids: list[int] = field(default_factory=list)  # 다시 잇기로 묶인 원래 Track 번호들
    box_conf_sum: float = 0.0   # 박스 신뢰도 합 (오래, 또렷하게 보일수록 큼) → 주인공 고르기에 사용
    is_primary: bool = False


@dataclass
class SkippedPerson:
    track_id: int
    valid_frames: int
    reason: str                 # 학습 때(34번)와 같은 이름: TOO_FEW_VALID_FRAMES
    merged_track_ids: list[int] = field(default_factory=list)


@dataclass
class VideoAnalysis:
    frame_count: int
    fps: float
    width: int
    height: int
    persons: list[PersonSequence] = field(default_factory=list)
    skipped: list[SkippedPerson] = field(default_factory=list)
    detected_track_ids: list[int] = field(default_factory=list)


def _row_from_keypoints(frame_id: int, xy: np.ndarray) -> dict:
    """26번이 CSV에 쓰던 한 줄과 같은 모양을 만든다.

    26번은 픽셀 좌표를 소수점 셋째 자리까지 반올림해 CSV에 저장했고,
    34번은 그 CSV를 읽어 계산했다. 같은 숫자를 만들려고 반올림까지 따라 한다.
    """
    row = {"frame_id": frame_id, "pose_detected": 1}
    for index, name in enumerate(COCO_KEYPOINT_NAMES):
        row[f"{name}_x_px"] = round(float(xy[index][0]), 3)
        row[f"{name}_y_px"] = round(float(xy[index][1]), 3)
    return row


def _select_person(result) -> int | None:
    """26번 select_person()과 같은 규칙: 박스 신뢰도가 가장 높은 사람."""
    if result.boxes is None or result.keypoints is None or result.keypoints.xy is None:
        return None
    if len(result.boxes) == 0 or len(result.keypoints.xy) == 0:
        return None
    box_conf = result.boxes.conf.detach().cpu().numpy()
    count = min(len(box_conf), len(result.keypoints.xy))
    if count == 0:
        return None
    return int(np.argmax(box_conf[:count]))


def _iou(a, b) -> float:
    """두 박스(x1, y1, x2, y2)가 겹치는 정도. 0 = 안 겹침, 1 = 완전히 같음."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _group_tracks(tracks: dict[int, dict[int, tuple]], max_gap_frames: int) -> list[list[int]]:
    """쪼개진 Track을 사람 단위 묶음으로 만든다. tracks[track_id][frame_id] = (box, conf, xy)"""
    parent = {t: t for t in tracks}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    ids = sorted(tracks)
    for i_pos, i in enumerate(ids):
        for j in ids[i_pos + 1:]:
            fi, fj = tracks[i], tracks[j]
            together = set(fi) & set(fj)
            if together:
                # ① 중복: 잠깐만 함께 보이고, 그동안 박스가 크게 겹침
                short = min(len(fi), len(fj))
                ious = [_iou(fi[f][0], fj[f][0]) for f in together]
                if len(together) / short <= DUPLICATE_MAX_OVERLAP_RATIO and float(np.median(ious)) >= DUPLICATE_MIN_IOU:
                    parent[find(j)] = find(i)
            else:
                # ② 끊김: 먼저 끝난 Track의 마지막 박스와 나중 Track의 첫 박스
                first, later = (fi, fj) if max(fi) < min(fj) else (fj, fi)
                gap = min(later) - max(first)
                if 0 < gap <= max_gap_frames and _iou(first[max(first)][0], later[min(later)][0]) >= RECONNECT_MIN_IOU:
                    parent[find(j)] = find(i)

    groups: dict[int, list[int]] = {}
    for t in ids:
        groups.setdefault(find(t), []).append(t)
    return [sorted(members) for members in groups.values()]


def extract_person_sequences(video_path: Path, bundle: ModelBundle, mode: str = "track") -> VideoAnalysis:
    """영상 파일 하나를 읽어 사람마다 1,140개 숫자를 만든다."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoReadError("영상을 열 수 없습니다. 파일이 손상되었거나 영상 형식이 아닙니다.")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cfg = bundle.project_config
    predict_args = dict(imgsz=int(cfg["image_size"]), conf=float(cfg["person_confidence"]),
                        device=bundle.device, verbose=False)

    if mode == "track":
        # 추적기는 프레임 사이의 기억(상태)을 가진다. 요청끼리 기억이 섞이지 않도록
        # 영상마다 YOLO를 새로 만든다 (서버 시작 때 불러 둔 모델을 함께 쓰면 두 요청의 Track이 섞일 수 있다).
        pose = YOLO(str(bundle.pose_weights))
        tracker = bundle.tracking_config.get("tracker", "bytetrack.yaml")
    elif mode == "single":
        pose = bundle.pose_model
    else:
        raise ValueError(f"알 수 없는 mode: {mode}")

    rows_by_person: dict[int, list[dict]] = {}
    tracks: dict[int, dict[int, tuple]] = {}   # track 모드: tracks[track_id][frame_id] = (box, conf, xy)
    frame_id = 0  # 26번과 같이 0부터 센다
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if mode == "track":
                result = pose.track(frame, persist=True, tracker=tracker, **predict_args)[0]
                ids = result.boxes.id if result.boxes is not None else None
                if ids is not None and result.keypoints is not None and result.keypoints.xy is not None:
                    track_ids = ids.int().cpu().tolist()
                    boxes = result.boxes.xyxy.detach().cpu().numpy()
                    confs = result.boxes.conf.detach().cpu().numpy()
                    all_xy = result.keypoints.xy.detach().cpu().numpy()
                    for k, track_id in enumerate(track_ids):
                        if all_xy[k].shape == (17, 2):
                            tracks.setdefault(int(track_id), {})[frame_id] = (boxes[k].tolist(), float(confs[k]), all_xy[k])
            else:
                result = pose.predict(source=frame, **predict_args)[0]
                index = _select_person(result)
                if index is not None:
                    xy = result.keypoints.xy[index].detach().cpu().numpy()
                    if xy.shape == (17, 2):
                        rows_by_person.setdefault(1, []).append(_row_from_keypoints(frame_id, xy))
            frame_id += 1
    finally:
        capture.release()

    if frame_id == 0:
        raise VideoReadError("영상에서 프레임을 하나도 읽지 못했습니다.")

    members_of: dict[int, list[int]] = {}
    conf_sum_of: dict[int, float] = {}
    if mode == "track":
        lost_seconds = float(bundle.tracking_config.get("lost_timeout_seconds", 2.0))
        max_gap = max(1, int(round(lost_seconds * (fps if fps > 0 else 25.0))))
        for members in _group_tracks(tracks, max_gap):
            person_id = members[0]           # 묶음의 대표 번호 = 가장 먼저 생긴 Track 번호
            frames: dict[int, tuple] = {}
            for t in members:
                for f, entry in tracks[t].items():
                    # 같은 프레임에 박스가 둘이면 신뢰도 1등만 남긴다 (26번과 같은 규칙)
                    if f not in frames or entry[1] > frames[f][1]:
                        frames[f] = entry
            rows_by_person[person_id] = [_row_from_keypoints(f, frames[f][2]) for f in sorted(frames)]
            members_of[person_id] = members
            conf_sum_of[person_id] = float(sum(e[1] for e in frames.values()))
    else:
        for person_id in rows_by_person:
            members_of[person_id] = [person_id]

    analysis = VideoAnalysis(frame_count=frame_id, fps=fps, width=width, height=height,
                             detected_track_ids=sorted(tracks) if mode == "track" else sorted(rows_by_person))

    for track_id in sorted(rows_by_person):
        frame_df = pd.DataFrame(rows_by_person[track_id])
        # ↓ 학습 때(34번)와 같은 함수, 같은 순서
        matrix, frame_ids = build_frame_feature_matrix(frame_df)
        if len(matrix) < bundle.min_valid_frames:
            analysis.skipped.append(SkippedPerson(track_id, int(len(matrix)), "TOO_FEW_VALID_FRAMES",
                                                  members_of.get(track_id, [track_id])))
            continue
        sequence = resample_sequence(matrix, frame_ids, target_steps=bundle.sequence_steps)
        vector = flatten_sequence(sequence)
        if len(vector) != bundle.sequence_steps * FEATURES_PER_STEP:
            raise RuntimeError("Sequence Feature 수가 예상과 다릅니다.")
        analysis.persons.append(PersonSequence(track_id, vector, int(len(matrix)),
                                               int(frame_ids[0]), int(frame_ids[-1]),
                                               members_of.get(track_id, [track_id]),
                                               conf_sum_of.get(track_id, 0.0)))

    # 주인공: 예측할 수 있는 사람 가운데 박스 신뢰도 합이 가장 큰 사람 (single 모드는 한 사람뿐)
    if analysis.persons:
        primary = max(analysis.persons, key=lambda p: (p.box_conf_sum, p.valid_frames))
        primary.is_primary = True
    return analysis
