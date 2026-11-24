"""Django Admin(/admin/)에 두 모델을 등록한다.
    관리자가 전체 작업과 사람별 결과를 보고 찾을 수 있게."""

from django.contrib import admin

from .models import AnalysisJob, PersonPrediction


class PersonPredictionInline(admin.TabularInline):
    """작업 상세 화면 아래에 그 작업의 사람별 예측을 표로 보여 준다."""

    model = PersonPrediction
    extra = 0
    fields = (
        "track_id",
        "is_primary",
        "action_label",
        "confidence",
        "is_uncertain",
        "input_quality",
        "valid_frames",
    )
    readonly_fields = fields
    can_delete = False


@admin.register(AnalysisJob)
class AnalysisJobAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "input_type",
        "status",
        "result_status",
        "expected_action",
        "created_at",
        "completed_at",
    )
    list_filter = (
        "status",
        "input_type",
        "created_at",
    )  # 상태별·입력 방식별 필터 (과제 9-2)
    search_fields = ("user__username", "title", "celery_task_id")  # 사용자 검색
    date_hierarchy = "created_at"
    readonly_fields = (
        "created_at",
        "started_at",
        "completed_at",
        "processing_ms",
        "celery_task_id",
    )
    inlines = [PersonPredictionInline]


@admin.register(PersonPrediction)
class PersonPredictionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "job",
        "track_id",
        "is_primary",
        "action_label",
        "confidence",
        "is_uncertain",
        "input_quality",
    )
    list_filter = ("action_label", "is_primary", "is_uncertain")
    search_fields = ("job__user__username",)
    list_select_related = ("job", "job__user")
