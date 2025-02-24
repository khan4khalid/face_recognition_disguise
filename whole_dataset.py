import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Load the trained model
class PatchClassifierCNN(nn.Module):
    def __init__(self):
        super(PatchClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        return x

model = PatchClassifierCNN()
model.load_state_dict(torch.load("cnn_patch_classifier.pth"))
model.eval()

# Transform for patches
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Function to divide the image into patches
def extract_patches(image, patch_size):
    patches = []
    positions = []
    width, height = image.size
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)
            positions.append((x, y))
    return patches, positions

# Function to classify patches and reconstruct the image
def classify_and_reconstruct(image_path, output_path, patch_size=32):
    image = Image.open(image_path).convert("RGB")
    patches, positions = extract_patches(image, patch_size)

    clean_patches = []
    clean_positions = []

    for patch, position in zip(patches, positions):
        patch_tensor = transform(patch).unsqueeze(0)

        with torch.no_grad():
            logits = model(patch_tensor)
            _, predicted = torch.max(logits, 1)

        if predicted.item() == 0:  # Class 0: Clean
            clean_patches.append(patch)
            clean_positions.append(position)

    # Reconstruct the image with clean patches
    reconstructed_image = Image.new("RGB", image.size)
    for patch, position in zip(clean_patches, clean_positions):
        reconstructed_image.paste(patch, position)

    # Dynamically construct the output file extension
    if not os.path.splitext(output_path)[1]:
        output_path += ".jpg"  # Default to .jpg if no extension is provided

    # Save the reconstructed image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    reconstructed_image.save(output_path)
    print(f"Reconstructed image saved at {output_path}")

# Process entire dataset
input_dataset_dir = "dataset"
output_dataset_dir = "clean_dataset"
patch_size = 32

for root, dirs, files in os.walk(input_dataset_dir):
    for file in files:
        if file.endswith(('.jpg', '.png', '.jpeg')):
            input_image_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, input_dataset_dir)
            output_image_path = os.path.join(output_dataset_dir, relative_path, file)

            classify_and_reconstruct(input_image_path, output_image_path, patch_size=patch_size)
