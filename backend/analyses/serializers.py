"""DRF Serializer: 들어오는 값을 검사하고, 나가는 값을 JSON 모양으로 바꾼다.

- 만들 때(Create): 입력 방식에 맞는 값이 왔는지 검사한다 (업무 규칙은 모델이 아니라 여기서)
- 볼 때(Read): 작업 + 사람별 예측을 함께 보여 준다
- 고칠 때(Update): 제목·메모만 받는다. 상태·예측값은 사용자가 바꿀 수 없다 (과제 3-2)
"""
from pathlib import Path

from django.conf import settings
from rest_framework import serializers

from .models import AnalysisJob, PersonPrediction


class PersonPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = PersonPrediction
        fields = ["track_id", "is_primary", "merged_track_ids", "action_label", "confidence", "is_uncertain",
                  "input_quality", "valid_frames", "first_frame", "last_frame", "window_count", "probabilities"]
        read_only_fields = fields


class AnalysisJobSerializer(serializers.ModelSerializer):
    """목록·상세 조회용. 사람별 예측을 안에 담아 보여 준다 (GET /api/v1/analyses/{id}/)."""
    predictions = PersonPredictionSerializer(many=True, read_only=True)
    status_display = serializers.CharField(source="get_status_display", read_only=True)
    user = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = AnalysisJob
        fields = ["id", "user", "title", "memo", "input_type", "video_file", "camera_name", "capture_seconds",
                  "expected_action", "status", "status_display", "result_status", "error_code", "error_message",
                  "processing_ms", "celery_task_id", "created_at", "started_at", "completed_at", "predictions"]
        read_only_fields = fields


class AnalysisJobCreateSerializer(serializers.ModelSerializer):
    """작업 만들기 (POST /api/v1/analyses/). 입력 방식에 따라 필요한 값이 다르다."""

    class Meta:
        model = AnalysisJob
        fields = ["id", "title", "memo", "input_type", "video_file", "camera_name", "capture_seconds",
                  "expected_action", "status"]
        read_only_fields = ["id", "status"]

    def validate_video_file(self, file):
        if file is None:
            return file
        suffix = Path(file.name).suffix.lower()
        if suffix not in settings.UPLOAD_ALLOWED_SUFFIXES:
            raise serializers.ValidationError(f"영상 파일만 올릴 수 있습니다 ({', '.join(settings.UPLOAD_ALLOWED_SUFFIXES)}).")
        if file.size == 0:
            raise serializers.ValidationError("빈 파일입니다.")
        if file.size > settings.UPLOAD_MAX_MB * 1024 * 1024:
            raise serializers.ValidationError(f"파일이 너무 큽니다 (최대 {settings.UPLOAD_MAX_MB}MB).")
        return file

    def validate(self, data):
        """칸끼리 맞는지 검사: 업로드면 파일 필수, IP Webcam이면 허용된 카메라 이름 필수."""
        input_type = data.get("input_type")
        if input_type == AnalysisJob.InputType.UPLOAD:
            if not data.get("video_file"):
                raise serializers.ValidationError({"video_file": "영상 업로드 방식은 영상 파일이 필요합니다."})
            if data.get("camera_name") or data.get("capture_seconds"):
                raise serializers.ValidationError({"camera_name": "영상 업로드 방식에는 카메라 값을 넣지 않습니다."})
        elif input_type == AnalysisJob.InputType.IP_WEBCAM:
            if data.get("video_file"):
                raise serializers.ValidationError({"video_file": "IP Webcam 방식에는 영상 파일을 넣지 않습니다."})
            name = data.get("camera_name", "")
            allowed = settings.IP_WEBCAM_CAMERAS
            # 사용자가 아무 주소나 넣게 하지 않는다: 서버에 등록된 카메라 이름만 받는다 (과제 8-1)
            if name not in allowed:
                choices = ", ".join(allowed) or "(등록된 카메라 없음)"
                raise serializers.ValidationError({"camera_name": f"허용된 카메라 이름이 아닙니다. 고를 수 있는 이름: {choices}"})
            seconds = data.get("capture_seconds") or settings.IP_WEBCAM_CAPTURE_SECONDS
            low, high = settings.IP_WEBCAM_CAPTURE_RANGE
            if not low <= seconds <= high:
                raise serializers.ValidationError({"capture_seconds": f"수집 시간은 {low}~{high}초입니다."})
            data["capture_seconds"] = seconds
        return data


class AnalysisJobUpdateSerializer(serializers.ModelSerializer):
    """고치기 (PATCH /api/v1/analyses/{id}/). 제목·메모만. 다른 칸을 보내면 거절한다."""

    class Meta:
        model = AnalysisJob
        fields = ["title", "memo"]

    def validate(self, data):
        extra = set(self.initial_data) - set(self.fields)
        if extra:
            raise serializers.ValidationError({name: "이 칸은 고칠 수 없습니다." for name in sorted(extra)})
        return data
