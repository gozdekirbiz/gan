from django.urls import path
from . import views
from .views import signup


urlpatterns = [
    path("", views.login_view, name="login"),
    path("signup/", signup, name="signup"),
    path("home/", views.home_view, name="home"),
    path("logout/", views.logout_view, name="logout"),
    # other paths...
]
