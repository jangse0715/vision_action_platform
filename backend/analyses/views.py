"""분석 작업 API (DRF).

    POST   /api/v1/analyses/        작업 만들기 → DB에 PENDING으로 저장 → 저장이 확정되면 Worker 대기열에 등록 (6단계)
    GET    /api/v1/analyses/        내 작업 목록
    GET    /api/v1/analyses/{id}/   내 작업 상세 = 상태 + 사람별 결과 (가이드: 상태 조회도 이 주소로)
    PATCH  /api/v1/analyses/{id}/   제목·메모만 고치기
    DELETE /api/v1/analyses/{id}/   내 작업 지우기 (사람별 예측·업로드 영상도 함께)

권한: 로그인한 사람만 (settings.REST_FRAMEWORK). 모든 동작에서 '내 작업'만 보인다.
      남의 작업 번호를 넣으면 403이 아니라 404 — 그런 작업이 있는지조차 알려 주지 않는다 (T07).
"""
from django.db import transaction
from rest_framework import status, viewsets
from rest_framework.response import Response

from .models import AnalysisJob
from .serializers import AnalysisJobCreateSerializer, AnalysisJobSerializer, AnalysisJobUpdateSerializer
from .tasks import queue_analysis_job


class AnalysisJobViewSet(viewsets.ModelViewSet):
    http_method_names = ["get", "post", "patch", "delete", "head", "options"]  # PUT(통째로 바꾸기)은 막는다

    def get_queryset(self):
        # 관리자라도 API에서는 자기 작업만. 전체 기록은 관리자 페이지·대시보드(10단계)에서 본다
        return (AnalysisJob.objects.filter(user=self.request.user)
                .select_related("user").prefetch_related("predictions"))

    def get_serializer_class(self):
        if self.action == "create":
            return AnalysisJobCreateSerializer
        if self.action == "partial_update":
            return AnalysisJobUpdateSerializer
        return AnalysisJobSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        # 요청한 사람은 로그인 정보로 정한다 (사용자가 user 칸을 보내서 남의 이름으로 만들 수 없게)
        job = serializer.save(user=request.user, status=AnalysisJob.Status.PENDING)
        queue_analysis_job(job)   # 6단계: 저장이 확정된 뒤 Redis 대기열에 등록 → Worker가 처리
        job.refresh_from_db()     # 등록하며 적은 Celery 작업 ID(또는 등록 실패 표시)를 응답에 담기 위해
        body = AnalysisJobSerializer(job, context=self.get_serializer_context()).data
        return Response(body, status=status.HTTP_201_CREATED)

    def partial_update(self, request, *args, **kwargs):
        response = super().partial_update(request, *args, **kwargs)
        # 고친 뒤에는 전체 모습(상태·결과 포함)을 돌려준다
        response.data = AnalysisJobSerializer(self.get_object(), context=self.get_serializer_context()).data
        return response

    def perform_destroy(self, job):
        video = job.video_file
        job.delete()  # 사람별 예측은 CASCADE로 함께 지워진다
        if video:
            # DB 삭제가 확정된 뒤에 파일도 지운다 (Django는 모델을 지워도 파일은 남겨 둔다)
            transaction.on_commit(lambda: video.storage.delete(video.name))
