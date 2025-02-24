'''database creation for clean dataset

import os
import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import transforms
import logging

# Configuration
PATCH_SIZE = (32, 32)  # Patch size
dataset_path = "clean_dataset"  # Main dataset path
feature_output_path = "feature_database"  # Output database path
os.makedirs(feature_output_path, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="feature_extraction.log",  # Log file
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler())  # Also log to console

# Load pre-trained ResNet50 model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # Updated weights
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove the final classification layer
model.eval()

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(PATCH_SIZE),  # Resize patches to model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize for ResNet
])

# Function to divide image into patches
def divide_image_into_patches(image, patch_size=PATCH_SIZE):
    """
    Divides an image into non-overlapping patches of the specified size.
    Skips black patches (if all pixel values are zero).
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size[1]):
        for j in range(0, width, patch_size[0]):
            patch = image.crop((j, i, j + patch_size[0], i + patch_size[1]))
            # Exclude black patches (all zero pixels)
            if np.array(patch).sum() > 0:  # Skip black (empty) patches
                patches.append(patch)
    return patches

# Function to extract features for a single patch
def extract_patch_feature(patch):
    patch_tensor = preprocess(patch).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        feature = model(patch_tensor).squeeze().numpy()  # Extract feature vector
    return feature

# Function to aggregate features (mean pooling)
def aggregate_features(features):
    return np.mean(features, axis=0)

# Process biometric images
def process_biometric(biometric_path, output_path):
    biometric_output_path = os.path.join(output_path, "biometric")
    os.makedirs(biometric_output_path, exist_ok=True)

    for person in os.listdir(biometric_path):
        person_path = os.path.join(biometric_path, person)
        person_features = []

        logging.info(f"Processing biometric folder: {person}")

        for image_file in os.listdir(person_path):
            image_path = os.path.join(person_path, image_file)
            logging.info(f"Processing image: {image_file}")

            image = Image.open(image_path)

            # Divide image into patches and extract features
            patches = divide_image_into_patches(image, patch_size=PATCH_SIZE)
            patch_features = [extract_patch_feature(patch) for patch in patches]
            
            # Aggregate features for the image
            image_feature = aggregate_features(patch_features)
            person_features.append(image_feature)

        # Aggregate features for the person
        aggregated_person_feature = aggregate_features(person_features)
        output_file = os.path.join(biometric_output_path, f"{person}_aggregated_clean_feature.npy")
        np.save(output_file, aggregated_person_feature)
        logging.info(f"Saved aggregated clean feature for {person} to {output_file}")

# Process non-biometric images
def process_non_biometric(non_biometric_path, output_path):
    non_biometric_output_path = os.path.join(output_path, "non_biometric")
    os.makedirs(non_biometric_output_path, exist_ok=True)

    for person in os.listdir(non_biometric_path):
        person_path = os.path.join(non_biometric_path, person)
        person_output_path = os.path.join(non_biometric_output_path, person)
        os.makedirs(person_output_path, exist_ok=True)

        logging.info(f"Processing non-biometric folder: {person}")

        for disguise_type in os.listdir(person_path):
            disguise_path = os.path.join(person_path, disguise_type)
            if not os.path.isdir(disguise_path):
                continue

            logging.info(f"Processing disguise type: {disguise_type}")

            disguise_features = []

            for image_file in os.listdir(disguise_path):
                image_path = os.path.join(disguise_path, image_file)
                logging.info(f"Processing image: {image_file}")

                image = Image.open(image_path)

                # Divide image into patches and extract features
                patches = divide_image_into_patches(image, patch_size=PATCH_SIZE)
                patch_features = [extract_patch_feature(patch) for patch in patches]

                # Aggregate features for the image
                image_feature = aggregate_features(patch_features)
                disguise_features.append(image_feature)

            # Aggregate features for the disguise type
            aggregated_disguise_feature = aggregate_features(disguise_features)
            output_file = os.path.join(person_output_path, f"{disguise_type}_feature.npy")
            np.save(output_file, aggregated_disguise_feature)
            logging.info(f"Saved feature for {person} {disguise_type} to {output_file}")

# Main function to process both datasets
def main():
    biometric_path = os.path.join(dataset_path, "biometric")
    non_biometric_path = os.path.join(dataset_path, "non_biometric")

    logging.info("Starting biometric data processing...")
    process_biometric(biometric_path, feature_output_path)

    logging.info("Starting non-biometric data processing...")
    process_non_biometric(non_biometric_path, feature_output_path)

if __name__ == "__main__":
    main()'''


'''for faster processing of database using multiprocessing
import os
import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import transforms
import logging
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import torch.multiprocessing as mp

# Configuration
PATCH_SIZE = (32, 32)  # Patch size
dataset_path = "clean_dataset"  # Main dataset path
feature_output_path = "feature_database"  # Output database path
os.makedirs(feature_output_path, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="feature_extraction.log",  # Log file
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler())  # Also log to console

# Load pre-trained ResNet50 model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # Updated weights
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove the final classification layer
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(PATCH_SIZE),  # Resize patches to model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize for ResNet
])

# Function to divide image into patches
def divide_image_into_patches(image, patch_size=PATCH_SIZE):
    """Divides an image into non-overlapping patches and returns them in batch.
        Returns a numpy array of patches, and a tuple of their positions.
    """
    patches = []
    positions = []
    width, height = image.size
    for i in range(0, height, patch_size[1]):
        for j in range(0, width, patch_size[0]):
            patch = image.crop((j, i, j + patch_size[0], i + patch_size[1]))
            # Exclude black patches (all zero pixels)
            if np.array(patch).sum() > 0:  # Skip black (empty) patches
                patches.append(preprocess(patch))
                positions.append((j,i))
    return torch.stack(patches).to(device), positions

# Function to extract features for a batch of patches
def extract_batch_features(patch_batch):
    with torch.no_grad():
        features = model(patch_batch).squeeze().cpu().numpy()  # Extract feature vectors
    return features

# Function to aggregate features (mean pooling)
def aggregate_features(features):
    return np.mean(features, axis=0)

def process_image(image_path):
    """Processes a single image, returns the aggregated feature"""
    image = Image.open(image_path)
    patches, _ = divide_image_into_patches(image, patch_size=PATCH_SIZE)
    if len(patches) > 0:
        patch_features = extract_batch_features(patches)
        image_feature = aggregate_features(patch_features)
    else:
      image_feature = None
    return image_feature

def process_biometric_person(person_path, output_path):
    """Processes a single biometric person, save the aggregated feature"""
    person_features = []
    for image_file in os.listdir(person_path):
        image_path = os.path.join(person_path, image_file)
        image_feature = process_image(image_path)
        if image_feature is not None:
            person_features.append(image_feature)
    if len(person_features) > 0:
      aggregated_person_feature = aggregate_features(person_features)
      output_file = os.path.join(output_path, f"{os.path.basename(person_path)}_aggregated_clean_feature.npy")
      np.save(output_file, aggregated_person_feature)
      logging.info(f"Saved aggregated clean feature for {os.path.basename(person_path)} to {output_file}")


def process_non_biometric_disguise(disguise_path, output_path):
      disguise_features = []
      for image_file in os.listdir(disguise_path):
          image_path = os.path.join(disguise_path, image_file)
          image_feature = process_image(image_path)
          if image_feature is not None:
            disguise_features.append(image_feature)
      if len(disguise_features) > 0:
        aggregated_disguise_feature = aggregate_features(disguise_features)
        output_file = os.path.join(output_path, f"{os.path.basename(disguise_path)}_feature.npy")
        np.save(output_file, aggregated_disguise_feature)
        logging.info(f"Saved feature for {os.path.basename(os.path.dirname(disguise_path))} {os.path.basename(disguise_path)} to {output_file}")

# Process biometric images
def process_biometric(biometric_path, output_path, max_workers):
    biometric_output_path = os.path.join(output_path, "biometric")
    os.makedirs(biometric_output_path, exist_ok=True)

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as executor:
      futures = []
      for person in os.listdir(biometric_path):
          person_path = os.path.join(biometric_path, person)
          futures.append(executor.submit(process_biometric_person, person_path, biometric_output_path))

      for future in tqdm(futures, desc="Processing biometric data"):
        future.result()

# Process non-biometric images
def process_non_biometric(non_biometric_path, output_path, max_workers):
    non_biometric_output_path = os.path.join(output_path, "non_biometric")
    os.makedirs(non_biometric_output_path, exist_ok=True)
    
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as executor:
        futures = []
        for person in os.listdir(non_biometric_path):
            person_path = os.path.join(non_biometric_path, person)
            person_output_path = os.path.join(non_biometric_output_path, person)
            os.makedirs(person_output_path, exist_ok=True)
            for disguise_type in os.listdir(person_path):
              disguise_path = os.path.join(person_path, disguise_type)
              if not os.path.isdir(disguise_path):
                  continue
              futures.append(executor.submit(process_non_biometric_disguise, disguise_path, person_output_path))

        for future in tqdm(futures, desc="Processing non-biometric data"):
            future.result()

# Main function to process both datasets
def main():
    # Set the multiprocessing start method to 'spawn'
    mp.set_start_method("spawn", force=True)
    biometric_path = os.path.join(dataset_path, "biometric")
    non_biometric_path = os.path.join(dataset_path, "non_biometric")

    max_workers = os.cpu_count() # or set to a specific number
    logging.info(f"Using {max_workers} workers.")

    logging.info("Starting biometric data processing...")
    process_biometric(biometric_path, feature_output_path, max_workers)

    logging.info("Starting non-biometric data processing...")
    process_non_biometric(non_biometric_path, feature_output_path, max_workers)

if __name__ == "__main__":
    main()'''






'''pipline working previously for resnet50model'''
import os
import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import transforms
import logging

# Configuration
PATCH_SIZE = (32, 32)  # Patch size
output_feature_path = "output_features"  # Path to save extracted features
os.makedirs(output_feature_path, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="feature_extraction.log",  # Log file
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler())  # Also log to console

# Load pre-trained ResNet50 model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # Updated weights
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove the final classification layer
model.eval()

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(PATCH_SIZE),  # Resize patches to model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize for ResNet
])

# Function to divide image into patches
def divide_image_into_patches(image, patch_size=PATCH_SIZE):
    """
    Divides an image into non-overlapping patches of the specified size.
    Skips black patches (if all pixel values are zero).
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size[1]):
        for j in range(0, width, patch_size[0]):
            patch = image.crop((j, i, j + patch_size[0], i + patch_size[1]))
            # Exclude black patches (all zero pixels)
            if np.array(patch).sum() > 0:  # Skip black (empty) patches
                patches.append(patch)
    return patches

# Function to extract features for a single patch
def extract_patch_feature(patch):
    patch_tensor = preprocess(patch).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        feature = model(patch_tensor).squeeze().numpy()  # Extract feature vector
    return feature

# Function to aggregate features (mean pooling)
def aggregate_features(features):
    return np.mean(features, axis=0)

# Function to process a single image
def process_single_image(image_path, output_folder):
    """
    Processes a single image to extract features and save them.
    Args:
        image_path (str): Path to the input image.
        output_folder (str): Path to the folder where features will be saved.
    """
    try:
        image = Image.open(image_path)
        patches = divide_image_into_patches(image, patch_size=PATCH_SIZE)
        patch_features = [extract_patch_feature(patch) for patch in patches]

        # Aggregate features for the image
        aggregated_feature = aggregate_features(patch_features)
        feature_filename = os.path.join(output_folder, f"{os.path.basename(image_path).split('.')[0]}_feature.npy")
        np.save(feature_filename, aggregated_feature)

        logging.info(f"Features extracted and saved to {feature_filename}")
        return feature_filename
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None

if __name__ == "__main__":
    # Test the script with a single image
    test_image_path = "test_image.jpg"  # Replace with your test image path
    feature_file = process_single_image(test_image_path, output_feature_path)
    if feature_file:
        logging.info(f"Feature file saved at: {feature_file}")


