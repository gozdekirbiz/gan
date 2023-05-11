import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
from django.shortcuts import render
from django.contrib.auth import authenticate
from django.contrib.auth import login
from django.shortcuts import redirect
from django.contrib.auth.forms import UserCreationForm


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.conv_2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.norm_1 = nn.BatchNorm2d(256)
        self.norm_2 = nn.BatchNorm2d(256)

    def forward(self, x):
        output = self.norm_2(self.conv_2(F.relu(self.norm_1(self.conv_1(x)))))
        return output + x  # ES


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3
        )
        self.norm_1 = nn.BatchNorm2d(64)

        # down-convolution #
        self.conv_2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
        )
        self.conv_3 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.norm_2 = nn.BatchNorm2d(128)

        self.conv_4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1
        )
        self.conv_5 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.norm_3 = nn.BatchNorm2d(256)

        # residual blocks #
        residualBlocks = []
        for l in range(8):
            residualBlocks.append(ResidualBlock())
        self.res = nn.Sequential(*residualBlocks)

        # up-convolution #
        self.conv_6 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.conv_7 = nn.ConvTranspose2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.norm_4 = nn.BatchNorm2d(128)

        self.conv_8 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.conv_9 = nn.ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.norm_5 = nn.BatchNorm2d(64)

        self.conv_10 = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3
        )

    def forward(self, x):
        x = F.relu(self.norm_1(self.conv_1(x)))

        x = F.relu(self.norm_2(self.conv_3(self.conv_2(x))))
        x = F.relu(self.norm_3(self.conv_5(self.conv_4(x))))

        x = self.res(x)
        x = F.relu(self.norm_4(self.conv_7(self.conv_6(x))))
        x = F.relu(self.norm_5(self.conv_9(self.conv_8(x))))

        x = self.conv_10(x)

        x = sigmoid(x)

        return x


# Load the generator model
generator_path = "ganapp/static/generator_release.pth"
generator = torch.load(generator_path)
G = generator["g_state_dict"]

# Create the generator model for inference
G_inference = Generator()
G_inference.load_state_dict(G)
G_inference.eval()


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
            image_file = request.FILES["image"]
            fs = FileSystemStorage()
            filename = fs.save(image_file.name, image_file)
            uploaded_image_url = fs.url(filename)

            # Load and preprocess the image
            image = Image.open(image_file)
            preprocess = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.ToTensor(),
                ]
            )
            input_tensor = preprocess(image)
            input_tensor = input_tensor.unsqueeze(0)

            # Generate the output image
            with torch.no_grad():
                output_tensor = G_inference(input_tensor)
            output_image = transforms.ToPILImage()(output_tensor.squeeze().cpu())

            # Save the input and output images in the database
            generated_image = GeneratedImage(
                input_image=image_file, output_image=output_image
            )
            generated_image.save()

            return render(
                request,
                "home.html",
                {
                    "form": form,
                    "input_image": uploaded_image_url,
                    "output_image": generated_image.output_image.url,
                },
            )
    else:
        form = ImageUploadForm()

        return render(request, "home.html", {"form": form})


def logout_view(request):
    logout(request)
    return redirect("login")


def signup(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("home")
    else:
        form = UserCreationForm()

    context = {"form": form}
    return render(request, "signup.html", context)
