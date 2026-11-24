"""4단계 확인: 회원가입·로그인·로그아웃·비로그인 차단·일반/관리자 구분.
실행: .venv\Scripts\python manage.py test accounts
(테스트는 진짜 DB가 아니라 Django가 잠깐 만드는 테스트용 DB에서 돌고 끝나면 지워진다)
"""
from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

GOOD_PW = "VisionTest!2026"


class AccountTests(TestCase):
    def test_signup_creates_user_and_logs_in(self):
        r = self.client.post(reverse("signup"), {"username": "userA", "password1": GOOD_PW, "password2": GOOD_PW})
        self.assertRedirects(r, reverse("home"))
        self.assertTrue(User.objects.filter(username="userA").exists())
        self.assertEqual(self.client.get(reverse("me")).status_code, 200)   # 가입 뒤 바로 로그인 상태

    def test_duplicate_username_rejected(self):
        User.objects.create_user("userA", password=GOOD_PW)
        r = self.client.post(reverse("signup"), {"username": "userA", "password1": GOOD_PW, "password2": GOOD_PW})
        self.assertEqual(r.status_code, 200)                                 # 다시 가입 화면 + 오류
        self.assertEqual(User.objects.filter(username="userA").count(), 1)

    def test_weak_or_mismatched_password_rejected(self):
        for p1, p2 in [("1234", "1234"), (GOOD_PW, GOOD_PW + "x")]:
            r = self.client.post(reverse("signup"), {"username": "userB", "password1": p1, "password2": p2})
            self.assertEqual(r.status_code, 200)
        self.assertFalse(User.objects.filter(username="userB").exists())

    def test_login_and_logout(self):
        User.objects.create_user("userA", password=GOOD_PW)
        r = self.client.post(reverse("login"), {"username": "userA", "password": GOOD_PW})
        self.assertRedirects(r, reverse("home"))
        r = self.client.post(reverse("logout"))                              # 로그아웃은 POST
        self.assertRedirects(r, reverse("home"))
        self.assertEqual(self.client.get(reverse("me")).status_code, 302)   # 다시 비로그인

    def test_anonymous_is_redirected_to_login(self):
        r = self.client.get(reverse("me"))
        self.assertRedirects(r, f"{reverse('login')}?next={reverse('me')}")

    def test_normal_user_cannot_open_admin_but_staff_can(self):
        User.objects.create_user("userA", password=GOOD_PW)
        User.objects.create_superuser("boss", password=GOOD_PW)
        self.client.login(username="userA", password=GOOD_PW)
        self.assertNotEqual(self.client.get("/admin/").status_code, 200)    # 일반 회원은 관리자 페이지 못 봄
        self.client.logout()
        self.client.login(username="boss", password=GOOD_PW)
        self.assertEqual(self.client.get("/admin/").status_code, 200)
