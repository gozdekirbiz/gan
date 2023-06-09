from django.urls import path
from . import views
from .views import signup
from .views import archive_view
from .views import delete_image

urlpatterns = [
    path("", views.login_view, name="login"),
    path("signup/", signup, name="signup"),
    path("home/", views.home_view, name="home"),
    path("logout/", views.logout_view, name="logout"),
    path("archive/", archive_view, name="archive"),
    path("delete_image/<int:image_id>/", delete_image, name="delete_image"),
    # ...
    # other paths...
]
