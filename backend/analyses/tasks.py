"""분석 작업을 Worker에게 맡기고(queue_analysis_job), Worker가 실행하는(run_analysis_job) 곳 (6단계).

상태 흐름:  PENDING(대기) → PROCESSING(처리 중) → COMPLETED(완료) 또는 FAILED(실패)

6단계 범위: 작업이 Worker까지 전달되고 상태가 바뀌는 것까지만.
            실제 분석(FastAPI 호출, 사람별 결과 저장)은 7단계에서 pretend_to_analyze() 자리에 넣는다.
"""

import logging
import time

from celery import shared_task
from django.db import transaction
from django.utils import timezone

from .models import AnalysisJob

logger = logging.getLogger(__name__)

STEP6_PRETEND_SECONDS = (
    3  # 6단계 임시: '처리 중' 상태를 눈으로 볼 수 있게 잠깐 기다린다 (7단계에서 지움)
)


def queue_analysis_job(job):
    """[Django 쪽] 작업 저장이 DB에 확정된 뒤 Redis 대기열에 쪽지를 넣는다.

    왜 '확정된 뒤'인가: 저장이 끝나기 전에 Worker가 먼저 꺼내 가면 DB에서 그 작업을 못 찾는다.
    Redis가 꺼져 있으면 요청이 멈추지 않도록 작업을 FAILED(QUEUE_UNAVAILABLE)로 남긴다.
    """

    def send_to_queue():
        try:
            sent = run_analysis_job.delay(job.pk)
        except Exception as error:  # Redis 연결 실패 등
            logger.error("작업 #%s 대기열 등록 실패: %s", job.pk, error)
            AnalysisJob.objects.filter(pk=job.pk).update(
                status=AnalysisJob.Status.FAILED,
                error_code="QUEUE_UNAVAILABLE",
                error_message="작업 대기열(Redis)에 연결하지 못했습니다.",
                completed_at=timezone.now(),
            )
            return
        AnalysisJob.objects.filter(pk=job.pk).update(celery_task_id=sent.id)
        logger.info("작업 #%s 대기열 등록 (Celery 작업 ID %s)", job.pk, sent.id)

    transaction.on_commit(send_to_queue)


@shared_task
def run_analysis_job(job_id):
    """[Worker 쪽] 대기열에서 꺼낸 작업 하나를 실행한다."""
    # PENDING인 작업만 PROCESSING으로 바꾼다. 바꾼 줄 수가 0이면 이미 누가 처리했거나 지워진 작업이다
    # (같은 쪽지가 두 번 배달돼도 두 번 실행하지 않게)
    claimed = AnalysisJob.objects.filter(
        pk=job_id, status=AnalysisJob.Status.PENDING
    ).update(status=AnalysisJob.Status.PROCESSING, started_at=timezone.now())
    if not claimed:
        logger.warning("작업 #%s 건너뜀: 없거나 이미 대기 상태가 아님", job_id)
        return

    logger.info("작업 #%s 처리 시작", job_id)
    try:
        pretend_to_analyze(job_id)
    except Exception as error:
        logger.exception("작업 #%s 실패", job_id)
        AnalysisJob.objects.filter(pk=job_id).update(
            status=AnalysisJob.Status.FAILED,
            error_code="WORKER_ERROR",
            error_message=str(error)[:500],
            completed_at=timezone.now(),
        )
        return

    AnalysisJob.objects.filter(pk=job_id).update(
        status=AnalysisJob.Status.COMPLETED, completed_at=timezone.now()
    )
    logger.info("작업 #%s 완료", job_id)


def pretend_to_analyze(job_id):
    """6단계 임시 자리. 7단계에서 'FastAPI에 영상 보내기 → 사람별 결과 저장'으로 바꾼다."""
    time.sleep(STEP6_PRETEND_SECONDS)
