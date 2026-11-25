r"""6단계 확인: 작업이 대기열에 등록되고, Worker가 상태를 바꾸는지 (Redis·Worker 없이 시험).
실행: .venv\Scripts\python manage.py test analyses

Redis 대신 가짜(mock)를 끼워서 '언제·무엇을 넣었는지'만 확인한다.
진짜 Redis·Worker로 하는 확인은 00_변경목록.md의 수동 검증 순서를 따른다.
"""
import shutil
import tempfile
from unittest import mock

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from rest_framework.test import APITestCase

from . import tasks
from .models import AnalysisJob

TEMP_MEDIA = tempfile.mkdtemp()
URL = "/api/v1/analyses/"


def fake_video():
    return SimpleUploadedFile("clip.mp4", b"\x00\x00\x00\x18ftypmp42 fake", content_type="video/mp4")


@override_settings(MEDIA_ROOT=TEMP_MEDIA)
class QueueAnalysisJobTests(APITestCase):
    """[Django 쪽] POST로 작업을 만들면 저장이 확정된 뒤 대기열에 한 번 등록된다."""

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TEMP_MEDIA, ignore_errors=True)

    def setUp(self):
        self.user = User.objects.create_user("userA", password="VisionTest!2026")
        self.client.force_authenticate(self.user)

    def post_upload(self):
        return self.client.post(URL, {"input_type": "UPLOAD", "video_file": fake_video()}, format="multipart")

    def test_job_is_queued_only_after_commit(self):
        with mock.patch.object(tasks.run_analysis_job, "delay") as fake_delay:
            fake_delay.return_value.id = "fake-task-id"
            # captureOnCommitCallbacks: '저장 확정 뒤에 할 일'을 모아 두었다가 블록 끝에서 실행한다
            with self.captureOnCommitCallbacks(execute=True) as after_commit:
                r = self.post_upload()
                fake_delay.assert_not_called()            # 저장 확정 전에는 아직 대기열에 넣지 않았다
        self.assertEqual(r.status_code, 201, r.data)
        self.assertEqual(len(after_commit), 1)
        fake_delay.assert_called_once_with(r.data["id"])  # 작업 번호만 쪽지에 적어 보낸다
        job = AnalysisJob.objects.get(pk=r.data["id"])
        self.assertEqual(job.celery_task_id, "fake-task-id")
        self.assertEqual(job.status, "PENDING")           # 등록만 했지 아직 Worker가 처리하지 않았다

    def test_redis_down_marks_job_failed(self):
        with mock.patch.object(tasks.run_analysis_job, "delay", side_effect=ConnectionError("redis down")):
            with self.captureOnCommitCallbacks(execute=True):
                r = self.post_upload()
        self.assertEqual(r.status_code, 201)
        job = AnalysisJob.objects.get(pk=r.data["id"])
        self.assertEqual(job.status, "FAILED")
        self.assertEqual(job.error_code, "QUEUE_UNAVAILABLE")


@mock.patch.object(tasks, "STEP6_PRETEND_SECONDS", 0)   # 시험에서는 기다리지 않는다
class RunAnalysisJobTests(TestCase):
    """[Worker 쪽] run_analysis_job이 상태를 PENDING → PROCESSING → COMPLETED/FAILED로 바꾼다."""

    def setUp(self):
        self.user = User.objects.create_user("userA", password="VisionTest!2026")
        self.job = AnalysisJob.objects.create(user=self.user, input_type="UPLOAD")

    def test_pending_job_becomes_completed(self):
        tasks.run_analysis_job(self.job.pk)
        self.job.refresh_from_db()
        self.assertEqual(self.job.status, "COMPLETED")
        self.assertIsNotNone(self.job.started_at)
        self.assertIsNotNone(self.job.completed_at)

    def test_job_runs_only_once(self):
        tasks.run_analysis_job(self.job.pk)
        first_completed_at = AnalysisJob.objects.get(pk=self.job.pk).completed_at
        tasks.run_analysis_job(self.job.pk)               # 같은 쪽지가 한 번 더 와도
        self.assertEqual(AnalysisJob.objects.get(pk=self.job.pk).completed_at, first_completed_at)

    def test_deleted_job_is_skipped_without_error(self):
        job_id = self.job.pk
        self.job.delete()
        tasks.run_analysis_job(job_id)                    # 오류 없이 건너뛴다
        self.assertFalse(AnalysisJob.objects.filter(pk=job_id).exists())

    def test_error_during_analysis_marks_failed(self):
        with mock.patch.object(tasks, "pretend_to_analyze", side_effect=RuntimeError("boom")):
            tasks.run_analysis_job(self.job.pk)
        self.job.refresh_from_db()
        self.assertEqual(self.job.status, "FAILED")
        self.assertEqual(self.job.error_code, "WORKER_ERROR")
        self.assertIn("boom", self.job.error_message)
