from django.db import models
from django.contrib.auth.models import User


class Photo(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to="photos/")
    created_at = models.DateTimeField(auto_now_add=True)


class GeneratedImage(models.Model):
    input_image = models.ImageField(upload_to="input_images/")
    output_image = models.ImageField(upload_to="output_images/")

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"GeneratedImage {self.id}"
