"""분석 작업(AnalysisJob)과 사람별 예측(PersonPrediction).

관계 (ERD)
    auth_user 1 ──< AnalysisJob 1 ──< PersonPrediction
    - 회원 1명은 분석 작업을 0개 이상 가진다
    - 분석 작업 1건은 사람별 예측을 0개 이상 가진다 (사람이 없거나 실패면 0건, 두 사람이면 2건)
"""

from django.conf import settings
from django.db import models

# 6일차 모델의 Class 이름 (model.classes_와 같은 이름)
ACTION_CHOICES = [
    ("bend_return", "허리 굽혔다 펴기"),
    ("leg_raise_lower", "한쪽 다리 들었다 내리기"),
    ("walk_turn_walk", "걷다가 방향 바꿔 걷기"),
    ("drink_return", "음료 마시고 돌아오기"),
    ("sit_stand", "앉았다 일어나기"),
    ("wave", "손 흔들기"),
]


class AnalysisJob(models.Model):
    """분석 요청 한 건. 누가, 어떤 입력으로, 지금 어떤 상태인지를 기록한다."""

    class InputType(models.TextChoices):
        UPLOAD = "UPLOAD", "영상 업로드"
        IP_WEBCAM = "IP_WEBCAM", "IP Webcam"

    class Status(models.TextChoices):
        # Celery 내부 상태(PENDING/STARTED/SUCCESS/FAILURE)와 헷갈리지 않게 다른 이름을 쓴다
        PENDING = "PENDING", "대기"
        PROCESSING = "PROCESSING", "처리 중"
        COMPLETED = "COMPLETED", "완료"
        FAILED = "FAILED", "실패"

    # FK: 작업을 요청한 회원. 회원이 지워지면 그 회원의 작업도 함께 지운다
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="analysis_jobs",
        verbose_name="요청 회원",
    )
    # 사용자가 고칠 수 있는 칸 (DRF에서 PATCH 허용 예정)
    title = models.CharField("제목", max_length=100, blank=True)
    memo = models.TextField("메모", blank=True)

    # 입력
    input_type = models.CharField("입력 방식", max_length=20, choices=InputType.choices)
    video_file = models.FileField(
        "업로드 영상", upload_to="uploads/%Y/%m/%d/", blank=True
    )  # 파일은 media/에, DB엔 경로만
    camera_name = models.CharField(
        "카메라 이름", max_length=50, blank=True
    )  # 허용 목록의 이름 (주소는 .env에만)
    capture_seconds = models.PositiveSmallIntegerField(
        "수집 시간(초)", null=True, blank=True
    )
    # 사용자가 "이 동작을 할 거예요"라고 고른 것. 검증된 정답이 아니다 (과제 4절 원칙 5)
    expected_action = models.CharField(
        "예상 동작", max_length=30, choices=ACTION_CHOICES, blank=True
    )

    # 처리 상태 (시스템만 바꾼다)
    status = models.CharField(
        "상태",
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
        db_index=True,
    )
    celery_task_id = models.CharField("Celery 작업 ID", max_length=100, blank=True)
    result_status = models.CharField(
        "추론 결과 구분", max_length=20, blank=True
    )  # FastAPI의 ok/no_person/too_short/low_quality
    error_code = models.CharField("오류 코드", max_length=50, blank=True)
    error_message = models.TextField("오류 내용", blank=True)
    processing_ms = models.PositiveIntegerField("추론 시간(ms)", null=True, blank=True)

    created_at = models.DateTimeField("요청 시각", auto_now_add=True)
    started_at = models.DateTimeField("시작 시각", null=True, blank=True)
    completed_at = models.DateTimeField("완료 시각", null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "-created_at"])
        ]  # "내 분석 목록"을 빠르게
        verbose_name = "분석 작업"
        verbose_name_plural = "분석 작업"

    def __str__(self):
        return f"#{self.pk} {self.user} {self.get_input_type_display()} {self.status}"


class PersonPrediction(models.Model):
    """분석 작업 한 건 안에서, 영상 속 한 사람에 대한 예측 한 건."""

    # FK: 이 예측이 나온 작업. 작업이 지워지면 예측도 함께 지운다 (작업 없이 예측만 남는 것은 의미가 없음)
    job = models.ForeignKey(
        AnalysisJob, on_delete=models.CASCADE, related_name="predictions"
    )

    # track_id는 이 작업(영상) 안에서만 쓰는 번호. 영구적인 사람 ID가 아니다 (과제 4절 원칙 3)
    track_id = models.PositiveIntegerField("Track ID")
    merged_track_ids = models.JSONField("묶인 Track 번호", default=list, blank=True)
    is_primary = models.BooleanField(
        "주인공", default=False
    )  # 박스 신뢰도 합이 가장 큰 사람

    action_label = models.CharField("예측 동작", max_length=30, choices=ACTION_CHOICES)
    confidence = models.FloatField("Confidence")
    is_uncertain = models.BooleanField("판단 불확실", default=False)
    input_quality = models.FloatField(
        "입력 품질(관절 신뢰도 평균)", null=True, blank=True
    )
    probabilities = models.JSONField("동작별 확률", default=dict, blank=True)

    valid_frames = models.PositiveIntegerField("유효 프레임 수")
    first_frame = models.PositiveIntegerField("첫 프레임", null=True, blank=True)
    last_frame = models.PositiveIntegerField("끝 프레임", null=True, blank=True)
    window_count = models.PositiveSmallIntegerField("처리한 Window 수", default=1)

    created_at = models.DateTimeField("예측 저장 시각", auto_now_add=True)

    class Meta:
        ordering = ["job", "-is_primary", "track_id"]
        constraints = [
            # 한 작업 안에서 같은 Track ID가 두 번 저장되지 않게
            models.UniqueConstraint(
                fields=["job", "track_id"], name="uniq_prediction_job_track"
            ),
        ]
        verbose_name = "사람별 예측"
        verbose_name_plural = "사람별 예측"

    def __str__(self):
        return f"job#{self.job_id} track{self.track_id} {self.action_label} {self.confidence:.2f}"
