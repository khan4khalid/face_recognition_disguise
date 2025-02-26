# Face Recognition Under Disguise  

## 📌 Project Overview  
This project focuses on face recognition under disguise conditions, aiming to identify individuals even when they wear disguises like glasses, masks, scarves, or beards. It combines biometric (facial) and non-biometric (disguise) feature extraction to improve recognition accuracy.  

## 🚀 Features  
- Dual dataset support: Clean (biometric) and disguised (non-biometric) images.  
- Feature extraction: Uses ResNet50 (from InsightFace) for embeddings.  
- Disguise-aware matching: Compares clean and disguised images using similarity metrics.  
- Real-time recognitio: Flask-based web app for live testing.  
- Continuous learning: Updates embeddings dynamically as new data is added.  

## 📂 Dataset Structure  
The dataset is split into two main folders:  

dataset
│── biometric/ # Clean face images
│── non_biometric/ # Disguised images categorized as:
│ ├── Beard/
│ ├── Cap/
│ ├── Cap_and_Scarf/
│ ├── Beard_and_Cap/
│ ├── Beard_and_Glasses/
│ ├── Mustache/
│ ├── Glasses/
│ ├── Glasses_and_Mask/
│ ├── Scarf/
│ ├── Scarf_and_Glasses/



## 🛠️ Implementation Details  

### 1️⃣ Feature Extraction 
- We use ResNet50 (from InsightFace) for feature embedding extraction.  
- Why ResNet50? It provides robust deep feature representations, optimized for face recognition.  
- The extracted embeddings are stored and used for matching.  

### 2️⃣ Face Matching  
- Cosine similarity is used to compare embeddings.  
- Threshold-based matching determines identity.  
- We compute match scores and analyze disguise effects.  

### 3️⃣ Web App (Flask)  
- A simple UI for uploading/testing images.  
- Real-time face recognition** via webcam.  

## 📦 Installation  

### 1️⃣ Clone the Repository 
```bash
git clone https://github.com/your-repo-name.git
cd your-repo-name

## Dataset Download

The dataset required for this project is too large to be stored in this repository.  
You can download it from the following link:

📂 [Download Dataset](https://drive.google.com/drive/folders/17nuDgFVW083eOWN4iCM7S3TTxOkF1jkE?usp=sharing) 

