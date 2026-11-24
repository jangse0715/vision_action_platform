"""회원가입과 내 정보 화면.

- 회원 표는 Django 기본 User(auth_user)를 그대로 쓴다 (과제 3-1 권장: 커스텀 회원 테이블을 만들지 않음)
- 가입 양식은 Django 기본 UserCreationForm: 아이디 중복 검사, 비밀번호 두 번 입력 확인,
  settings.AUTH_PASSWORD_VALIDATORS(너무 짧음·흔함·숫자로만 등) 검사를 이미 갖고 있다
"""
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import redirect, render


def signup(request):
    if request.user.is_authenticated:          # 이미 로그인했다면 가입 화면이 필요 없다
        return redirect("home")
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():                    # 중복 아이디·비밀번호 규칙을 여기서 검사
            user = form.save()                 # auth_user 표에 한 줄 저장 (비밀번호는 해시로)
            login(request, user)               # 가입하자마자 로그인 상태로
            return redirect("home")
    else:
        form = UserCreationForm()
    return render(request, "accounts/signup.html", {"form": form})


@login_required                                # 비로그인이면 settings.LOGIN_URL(로그인 화면)로 보낸다
def me(request):
    return render(request, "accounts/me.html")
