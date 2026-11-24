"""URL 연결. 4단계: 관리자 페이지 + 첫 화면 + 회원(가입·로그인·로그아웃·내 정보). API·분석 화면은 다음 단계에서 붙인다."""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from django.views.generic import TemplateView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", TemplateView.as_view(template_name="home.html"), name="home"),
    # 회원가입·내 정보 (직접 만든 accounts 앱)
    path("accounts/", include("accounts.urls")),
    # 로그인·로그아웃 등 Django 기본 인증 화면 (accounts/login/, accounts/logout/ ...)
    path("accounts/", include("django.contrib.auth.urls")),
]

# 개발 중(DEBUG)에는 업로드된 영상 파일을 /media/ 주소로 볼 수 있게
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
