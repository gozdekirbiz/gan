import tempfile
from django.shortcuts import render
from django.contrib.auth import authenticate
from django.contrib.auth import login
from django.shortcuts import redirect
from django.contrib.auth.forms import UserCreationForm
from django import forms
from django.contrib.auth.models import User
from django.contrib import messages
from django.core.files.base import ContentFile

from ganapp.authentication import send_verification_email, generate_verification_code
from ganapp.forms import ImageUploadForm
from ganapp.generator import Cartoonize
from ganapp.models import UserImage

import os
import matplotlib.pyplot as plt
import numpy as np


def login_view(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            if user.is_superuser:
                return redirect("admin:index")
            else:
                return redirect("home")
        else:
            # Invalid login
            return render(
                request, "login.html", {"error": "Invalid username or password"}
            )
    else:
        return render(request, "login.html")


from django.contrib.auth.decorators import login_required

from django import forms


class ImageUploadForm(forms.Form):
    image = forms.ImageField(label="Upload Image")


@login_required
def home_view(request):
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            cartoonizer = Cartoonize()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
                temp.write(request.FILES["image"].read())
                cartoon_image = cartoonizer.forward(temp.name)
                # Save the original image and the cartoonized image to the database
                user_image = UserImage(
                    user=request.user,
                    image=request.FILES["image"],
                    output_image=cartoon_image,
                )  # Use image_path here
                user_image.save()
    else:
        form = ImageUploadForm()

    user_images = UserImage.objects.filter(user=request.user).last()
    return render(request, "home.html", {"form": form, "user_images": user_images})


def logout_view(request):
    logout(request)
    return redirect("login")


class SignUpForm(UserCreationForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")


def signup(request):
    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)

            # Doğrulama kodu oluştur
            verification_code = generate_verification_code()

            # Doğrulama kodunu e-posta ile gönder
            send_verification_email(user.email, verification_code)
            return redirect("home")
        else:
            error_occured = False
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, error)
                    error_occured = True
                    break
                if error_occured == True:
                    break
            # form hatalıysa, kullanıcının doldurduğu değerleri formda göster
            context = {"form": form}
    else:
        form = SignUpForm()

    context = {"form": form}
    return render(request, "signup.html", context)


from django.shortcuts import render
from .models import UserImage


@login_required
def archive_view(request):
    # Retrieve all the generated images for the current user
    user_images = UserImage.objects.filter(user=request.user)
    return render(request, "archive.html", {"user_images": user_images})


from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import UserImage
from django.contrib.auth.decorators import login_required

from django.shortcuts import get_object_or_404
from django.http import JsonResponse


from django.http import JsonResponse


def delete_image(request, image_id):
    if request.method == "POST":
        image = get_object_or_404(UserImage, id=image_id)
        image.delete()
        return JsonResponse({"success": True})
    else:
        return JsonResponse({"success": False})
