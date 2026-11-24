"""분석 앱 주소. 5단계는 API만. 분석 화면·대시보드 주소는 9·10단계에서 더한다."""
from rest_framework.routers import DefaultRouter

from .views import AnalysisJobViewSet

router = DefaultRouter()
router.register("analyses", AnalysisJobViewSet, basename="analysis")  # → /api/v1/analyses/, /api/v1/analyses/{id}/

urlpatterns = router.urls
