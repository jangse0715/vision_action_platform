r"""5단계 확인: 분석 작업 API (만들기·목록·상세·고치기·지우기, 입력 검사, 남의 작업 차단 T07).
실행: .venv\Scripts\python manage.py test analyses
"""
import shutil
import tempfile

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from rest_framework.test import APITestCase

from .models import AnalysisJob, PersonPrediction

TEMP_MEDIA = tempfile.mkdtemp()
URL = "/api/v1/analyses/"


def fake_video(name="clip.mp4"):
    return SimpleUploadedFile(name, b"\x00\x00\x00\x18ftypmp42 fake", content_type="video/mp4")


@override_settings(MEDIA_ROOT=TEMP_MEDIA, IP_WEBCAM_CAMERAS={"phone1": "http://192.168.0.10:8080/video"})
class AnalysisApiTests(APITestCase):
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TEMP_MEDIA, ignore_errors=True)

    def setUp(self):
        self.a = User.objects.create_user("userA", password="VisionTest!2026")
        self.b = User.objects.create_user("userB", password="VisionTest!2026")
        self.client.force_authenticate(self.a)

    # --- 만들기 ---
    def test_create_upload_job_is_pending_and_owned_by_me(self):
        r = self.client.post(URL, {"input_type": "UPLOAD", "video_file": fake_video(), "expected_action": "wave"},
                             format="multipart")
        self.assertEqual(r.status_code, 201, r.data)
        self.assertEqual(r.data["status"], "PENDING")
        job = AnalysisJob.objects.get(pk=r.data["id"])
        self.assertEqual(job.user, self.a)
        self.assertTrue(job.video_file.name.startswith("uploads/"))   # 파일은 media/에, DB엔 경로만

    def test_cannot_create_in_someone_elses_name(self):
        r = self.client.post(URL, {"input_type": "UPLOAD", "video_file": fake_video(), "user": self.b.pk},
                             format="multipart")
        self.assertEqual(r.status_code, 201)
        self.assertEqual(AnalysisJob.objects.get(pk=r.data["id"]).user, self.a)

    def test_upload_without_file_rejected(self):
        r = self.client.post(URL, {"input_type": "UPLOAD"}, format="multipart")
        self.assertEqual(r.status_code, 400)
        self.assertIn("video_file", r.data)

    def test_non_video_file_rejected(self):
        r = self.client.post(URL, {"input_type": "UPLOAD", "video_file": fake_video("notes.txt")}, format="multipart")
        self.assertEqual(r.status_code, 400)

    def test_ip_webcam_needs_allowed_camera_and_range(self):
        ok = self.client.post(URL, {"input_type": "IP_WEBCAM", "camera_name": "phone1"}, format="json")
        self.assertEqual(ok.status_code, 201, ok.data)
        self.assertEqual(ok.data["capture_seconds"], 8)               # 비우면 기본 8초
        bad = self.client.post(URL, {"input_type": "IP_WEBCAM", "camera_name": "http://evil.example/video"}, format="json")
        self.assertEqual(bad.status_code, 400)
        long = self.client.post(URL, {"input_type": "IP_WEBCAM", "camera_name": "phone1", "capture_seconds": 30}, format="json")
        self.assertEqual(long.status_code, 400)

    # --- 목록·상세 (T07) ---
    def test_list_shows_only_my_jobs(self):
        AnalysisJob.objects.create(user=self.a, input_type="IP_WEBCAM", camera_name="phone1")
        AnalysisJob.objects.create(user=self.b, input_type="IP_WEBCAM", camera_name="phone1")
        r = self.client.get(URL)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data["count"], 1)
        self.assertEqual(r.data["results"][0]["user"], "userA")

    def test_detail_includes_predictions(self):
        job = AnalysisJob.objects.create(user=self.a, input_type="UPLOAD", status="COMPLETED")
        PersonPrediction.objects.create(job=job, track_id=1, is_primary=True, action_label="wave",
                                        confidence=0.63, valid_frames=126)
        r = self.client.get(f"{URL}{job.pk}/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data["status"], "COMPLETED")
        self.assertEqual(r.data["predictions"][0]["action_label"], "wave")

    def test_other_users_job_is_404_for_read_update_delete(self):
        other = AnalysisJob.objects.create(user=self.b, input_type="IP_WEBCAM", camera_name="phone1")
        self.assertEqual(self.client.get(f"{URL}{other.pk}/").status_code, 404)
        self.assertEqual(self.client.patch(f"{URL}{other.pk}/", {"title": "x"}, format="json").status_code, 404)
        self.assertEqual(self.client.delete(f"{URL}{other.pk}/").status_code, 404)
        self.assertTrue(AnalysisJob.objects.filter(pk=other.pk).exists())

    # --- 고치기 ---
    def test_patch_title_and_memo_only(self):
        job = AnalysisJob.objects.create(user=self.a, input_type="IP_WEBCAM", camera_name="phone1")
        ok = self.client.patch(f"{URL}{job.pk}/", {"title": "손 흔들기 시험", "memo": "1차"}, format="json")
        self.assertEqual(ok.status_code, 200, ok.data)
        self.assertEqual(ok.data["title"], "손 흔들기 시험")
        bad = self.client.patch(f"{URL}{job.pk}/", {"status": "COMPLETED"}, format="json")
        self.assertEqual(bad.status_code, 400)
        job.refresh_from_db()
        self.assertEqual(job.status, "PENDING")                        # 상태는 사용자가 못 바꾼다
        self.assertEqual(self.client.put(f"{URL}{job.pk}/", {"title": "x"}, format="json").status_code, 405)

    # --- 지우기 ---
    def test_delete_removes_job_and_predictions(self):
        job = AnalysisJob.objects.create(user=self.a, input_type="UPLOAD")
        PersonPrediction.objects.create(job=job, track_id=1, action_label="wave", confidence=0.6, valid_frames=10)
        self.assertEqual(self.client.delete(f"{URL}{job.pk}/").status_code, 204)
        self.assertFalse(PersonPrediction.objects.filter(job_id=job.pk).exists())

    # --- 비로그인 ---
    def test_anonymous_is_rejected(self):
        self.client.force_authenticate(None)
        self.assertIn(self.client.get(URL).status_code, (401, 403))
        self.assertIn(self.client.post(URL, {"input_type": "UPLOAD"}).status_code, (401, 403))
