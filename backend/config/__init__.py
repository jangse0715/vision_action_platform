# Django가 켜질 때 Celery 앱도 함께 불러온다 (6단계).
# 그래야 views.py에서 작업을 등록할 때 이 프로젝트의 Celery 설정(Redis 주소 등)을 쓴다.
from .celery import app as celery_app

__all__ = ("celery_app",)
