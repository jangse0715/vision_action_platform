"""Celery 앱 (6단계).

Celery = 오래 걸리는 일을 웹 서버 밖에서 따로 처리하는 일꾼(Worker) 프로그램.
흐름: Django가 Redis 대기열에 '할 일 쪽지'를 넣는다 → Worker가 꺼내서 실행한다.

이 파일이 하는 일 세 가지
  1. Worker가 Django 설정(config/settings.py)을 쓰도록 알려 준다
  2. settings.py에서 CELERY_ 로 시작하는 값만 골라 Celery 설정으로 읽는다
  3. 각 앱의 tasks.py를 찾아 '할 수 있는 일' 목록에 올린다

Worker 실행 (backend 폴더에서, Windows):
    .venv\\Scripts\\celery -A config worker --loglevel=info --pool=solo
    --pool=solo : Windows는 Celery 기본 방식(prefork)을 지원하지 않아 한 번에 하나씩 처리하는 방식으로 띄운다
"""
import os

from celery import Celery

# Django 설정 파일 위치. manage.py에 적힌 것과 같은 값이다
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("config")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
