import matplotlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid

from django.core.files import File
from PIL import Image as PilImage
from django.core.files.base import ContentFile
from io import BytesIO
import torch
from torchvision import transforms

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

class Cartoonize():
    def __init__(self):

        # Load the generator model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generator_path = "ganapp/static/best_checkpoint.pth"

        # Load the whole state_dict
        loaded_state = torch.load(generator_path)

        # Get only the model state_dict
        model_state_dict = loaded_state['g_state_dict']

        # Assuming your generator is an instance of some `Generator` class
        self.generator = Generator()
        self.generator.load_state_dict(model_state_dict)
        self.generator = self.generator.to(self.device)
        self.generator.eval()

        self.transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def forward(self, image_path):

        image = PilImage.open(image_path)
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        output = self.generator(image).detach().cpu().clamp(0, 255).numpy()
        output_image = np.transpose(output[0], (1, 2, 0))
        plt.imsave('test.png', output_image)
        
        output_image = (output_image * 255).astype(np.uint8)
        output_image = PilImage.fromarray(output_image.astype('uint8'))  # Convert the numpy array to a PIL Image

        output_io = BytesIO()
        output_image.save(output_io, format='PNG')     
        return ContentFile(output_io.getvalue(), 'cartoonized.png')
        