import os
from pathlib import Path

from dotenv import load_dotenv

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent  # backend 폴더
PROJECT_ROOT = BASE_DIR.parent  # vision_action_platform 폴더

# 프로젝트 맨 위의 .env를 읽어 os.environ에 넣는다 (비밀값을 코드에 적지 않기 위해)
load_dotenv(PROJECT_ROOT / ".env")


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/6.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "change-me")

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.environ.get("DJANGO_DEBUG", "0") == "1"

ALLOWED_HOSTS = [
    h.strip()
    for h in os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    if h.strip()
]


# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # 설치해서 쓰는 패키지
    "rest_framework",
    # 직접 만든 앱
    "analyses",
    "accounts",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],  # backend/templates/ 의 HTML을 찾는다
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"


# Database
# https://docs.djangoproject.com/en/6.1/ref/settings/#databases

# PostgreSQL 접속 정보는 .env에서 읽는다 (3단계에서는 Docker로 띄운 PostgreSQL에 localhost:5432로 접속)
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("POSTGRES_DB", "vision_action"),
        "USER": os.environ.get("POSTGRES_USER", "vision"),
        "PASSWORD": os.environ.get("POSTGRES_PASSWORD", ""),
        "HOST": os.environ.get("POSTGRES_HOST", "localhost"),
        "PORT": os.environ.get("POSTGRES_PORT", "5432"),
    }
}


# Password validation
# https://docs.djangoproject.com/en/6.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/6.1/topics/i18n/

LANGUAGE_CODE = "ko-kr"
TIME_ZONE = "Asia/Seoul"

USE_I18N = True
USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/6.1/howto/static-files/

STATIC_URL = "static/"

# 업로드된 영상이 저장되는 곳. DB에는 파일 자체가 아니라 이 안의 경로만 저장한다 (과제 4절 원칙 4)
MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"


# Email
# https://docs.djangoproject.com/en/6.1/topics/email/#topic-email-configuration

MAILERS = {
    "default": {
        "BACKEND": "django.core.mail.backends.console.EmailBackend",
    },
}

# 로그인·로그아웃 (4단계)
LOGIN_URL = "login"                # 로그인이 필요한 화면에 비로그인으로 오면 보내는 곳
LOGIN_REDIRECT_URL = "home"        # 로그인 뒤 갈 곳
LOGOUT_REDIRECT_URL = "home"       # 로그아웃 뒤 갈 곳

# DRF API 설정 (5단계)
REST_FRAMEWORK = {
    # 누구인지 알아내는 방법: 브라우저 로그인 세션(웹 화면·Browsable API) + 아이디·비밀번호(curl·Postman 시험용)
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.BasicAuthentication",
    ],
    # 기본값: 로그인한 사람만 API를 쓸 수 있다
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.IsAuthenticated"],
    # 목록은 20개씩 나눠서
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 20,
}

# 분석 입력 규칙 (5단계 검사에 쓰고, 7·8단계 실행에 쓴다)
UPLOAD_MAX_MB = int(os.environ.get("UPLOAD_MAX_MB", "100"))
UPLOAD_ALLOWED_SUFFIXES = [".mp4", ".avi", ".mov", ".mkv"]
# 허용된 카메라: .env의 IP_WEBCAM_CAMERAS="이름=주소,이름=주소". DB에는 이름만 저장한다
IP_WEBCAM_CAMERAS = dict(
    item.split("=", 1) for item in os.environ.get("IP_WEBCAM_CAMERAS", "").split(",") if "=" in item
)
IP_WEBCAM_CAPTURE_SECONDS = int(os.environ.get("IP_WEBCAM_CAPTURE_SECONDS", "8"))
IP_WEBCAM_CAPTURE_RANGE = (5, 10)  # 과제: '5~10초 수집'
