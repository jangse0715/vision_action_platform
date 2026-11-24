"""회원 기능 주소. 로그인·로그아웃은 Django 기본 화면(config/urls.py의 django.contrib.auth.urls)을 쓴다."""
from django.urls import path

from . import views

urlpatterns = [
    path("signup/", views.signup, name="signup"),   # 회원가입
    path("me/", views.me, name="me"),               # 내 정보 (로그인한 사람만)
]
