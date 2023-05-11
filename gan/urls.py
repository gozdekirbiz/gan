from django.contrib import admin
from django.urls import include, path
from django.conf import settings
from django.conf.urls.static import static
from ganapp.views import signup

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("ganapp.urls")),
    path("signup/", signup, name="signup"),  # include the URLs from your app
    # other paths...
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
