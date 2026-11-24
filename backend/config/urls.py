"""URL 연결. 3단계에서는 관리자 페이지만 연다 (회원·API·화면 주소는 다음 단계에서 붙인다)."""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path

urlpatterns = [
    path("admin/", admin.site.urls),
]

# 개발 중(DEBUG)에는 업로드된 영상 파일을 /media/ 주소로 볼 수 있게
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
