import os
import shutil
import random
from torchvision import transforms
from PIL import Image

# Paths
CLEAN_DIR = "classified_patches/clean"
DISGUISED_DIR = "classified_patches/disguised"
BALANCED_CLEAN_DIR = "balanced_dataset/clean"
BALANCED_DISGUISED_DIR = "balanced_dataset/disguised"
os.makedirs(BALANCED_CLEAN_DIR, exist_ok=True)
os.makedirs(BALANCED_DISGUISED_DIR, exist_ok=True)

# Load clean and disguised patch file paths
clean_patches = [os.path.join(CLEAN_DIR, f) for f in os.listdir(CLEAN_DIR) if f.endswith(('.jpg', '.png'))]
disguised_patches = [os.path.join(DISGUISED_DIR, f) for f in os.listdir(DISGUISED_DIR) if f.endswith(('.jpg', '.png'))]

# Check class imbalance
print(f"Number of clean patches: {len(clean_patches)}")
print(f"Number of disguised patches: {len(disguised_patches)}")

# Augmentation for clean patches
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=10, scale=(0.9, 1.1)),
    transforms.ToTensor()
])

def augment_and_save(image_path, save_dir, augmentations, num_augmentations):
    """Apply augmentations and save the augmented images."""
    image = Image.open(image_path).convert("RGB")
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for i in range(num_augmentations):
        augmented_image = augmentations(image)
        augmented_image = transforms.ToPILImage()(augmented_image)
        augmented_image.save(os.path.join(save_dir, f"{base_name}_aug{i}.jpg"))

# Copy disguised patches to balanced dataset
for file_path in disguised_patches:
    shutil.copy(file_path, BALANCED_DISGUISED_DIR)

# Augment clean patches to balance the dataset
num_clean = len(clean_patches)
num_disguised = len(disguised_patches)

if num_clean < num_disguised:
    augmentation_factor = (num_disguised - num_clean) // num_clean + 1
    print(f"Augmenting clean patches with {augmentation_factor} augmentations per image...")

    for file_path in clean_patches:
        # Copy original clean image
        shutil.copy(file_path, BALANCED_CLEAN_DIR)
        # Generate augmented images
        augment_and_save(file_path, BALANCED_CLEAN_DIR, augmentation, augmentation_factor)

    # Remove excess clean patches if oversampled
    balanced_clean = len(os.listdir(BALANCED_CLEAN_DIR))
    if balanced_clean > num_disguised:
        excess = balanced_clean - num_disguised
        print(f"Removing {excess} excess clean patches...")
        clean_files = os.listdir(BALANCED_CLEAN_DIR)
        random.shuffle(clean_files)
        for file in clean_files[:excess]:
            os.remove(os.path.join(BALANCED_CLEAN_DIR, file))
else:
    print("Clean patches already balanced or in majority.")
    for file_path in clean_patches:
        shutil.copy(file_path, BALANCED_CLEAN_DIR)

# Verify the balanced dataset
balanced_clean = len(os.listdir(BALANCED_CLEAN_DIR))
balanced_disguised = len(os.listdir(BALANCED_DISGUISED_DIR))
print(f"Balanced clean patches: {balanced_clean}")
print(f"Balanced disguised patches: {balanced_disguised}")

