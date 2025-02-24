'''from flask import Flask, request, jsonify, render_template
import os
import torch
import numpy as np
from PIL import Image
import logging
from disguise_detection import detect_disguise  # Import detect_disguise directly
from test_classify import classify_and_reconstruct  # Import patch classification and reconstruction
from feature_extraction import process_single_image  # Import feature extraction for a single image
from matching import match_features  # Import matching logic

# Paths
input_image_folder = "input_images"  # Folder to save input images
biometric_path = "feature_database/biometric"
non_biometric_path = "feature_database/non_biometric"
output_clean_path = "output_clean_images"
output_features_path = "output_features"  # Folder to save extracted features

# Ensure required directories exist
os.makedirs(input_image_folder, exist_ok=True)
os.makedirs(output_clean_path, exist_ok=True)
os.makedirs(output_features_path, exist_ok=True)

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Flask App
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    if not file:
        return render_template('index.html', error="No file uploaded.")

    # Save the uploaded file to the input folder
    image_path = os.path.join(input_image_folder, file.filename)
    file.save(image_path)

    try:
        # Step 1: Disguise Detection
        disguise_name = detect_disguise(image_path)
        logging.info(f"Detected disguise: {disguise_name}")

        # Step 2: Patch Classification and Reconstruction
        output_image_path = os.path.join(output_clean_path, f"clean_{file.filename}")
        classify_and_reconstruct(image_path, output_image_path, patch_size=32)

        feature_file = process_single_image(output_image_path, output_features_path)
        if feature_file:
            query_feature = np.load(feature_file)
        else:
            logging.error(f"Feature extraction failed for image: {output_image_path}")
            return jsonify({"error": "Feature extraction failed. Please try again."}), 500


        # Step 4: Matching
        disguise_type = 0 if disguise_name == "clean" else 1
        best_match, similarity = match_features(query_feature, disguise_type, biometric_path, non_biometric_path)

        return render_template('index.html', result={
            "disguise": disguise_name,
            "best_match": best_match,
            "similarity": f"{similarity * 100:.2f}",
            "clean_image": f"/output_clean_images/{os.path.basename(output_image_path)}"
        })

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return render_template('index.html', error="An error occurred during processing. Please try again.")



@app.route('/add', methods=['POST'])
def add_to_database():
    file = request.files['image']
    label = request.form['label']
    database_type = request.form['type']  # biometric or non_biometric

    db_path = biometric_path if database_type == "biometric" else non_biometric_path

    # Save the uploaded image
    image_path = os.path.join(input_image_folder, file.filename)
    file.save(image_path)

    try:
        # Extract features
        feature_file = process_single_image(image_path, output_features_path)
        if not feature_file:
            return jsonify({"error": "Feature extraction failed"}), 500

        feature = np.load(feature_file)
        person_path = os.path.join(db_path, label)
        os.makedirs(person_path, exist_ok=True)
        feature_save_path = os.path.join(person_path, f"{file.filename.split('.')[0]}_feature.npy")
        np.save(feature_save_path, feature)

        return jsonify({"message": "Image added to database successfully.", "feature_file": feature_save_path})
    except Exception as e:
        logging.error(f"Error adding image to database: {e}")
        return jsonify({"error": "An error occurred while adding the image to the database."}), 500

if __name__ == '__main__':
    app.run(debug=True)
'''


'''from flask import Flask, request, jsonify, render_template
import os
import torch
import numpy as np
from PIL import Image
import logging
import base64
from io import BytesIO
import time  # Fix for unique filenames
from disguise_detection import detect_disguise
from test_classify import classify_and_reconstruct
from feature_extraction import process_single_image
from matching import match_features

# Paths
input_image_folder = "input_images"
biometric_path = "feature_database/biometric"
non_biometric_path = "feature_database/non_biometric"
output_clean_path = "output_clean_images"
output_features_path = "output_features"

# Ensure directories exist
os.makedirs(input_image_folder, exist_ok=True)
os.makedirs(output_clean_path, exist_ok=True)
os.makedirs(output_features_path, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Flask App
app = Flask(__name__, static_url_path='', static_folder='.')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'image' in request.files:  # File upload
            file = request.files['image']
            filename = f"{int(time.time())}_{file.filename}"
            image_path = os.path.join(input_image_folder, filename)
            file.save(image_path)
        elif request.json and 'image' in request.json:  # Webcam capture
            image_data = request.json['image'].split(',')[1]
            image_data = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_data))
            filename = "webcam_image.jpg"
            image_path = os.path.join(input_image_folder, filename)
            image.save(image_path)
        else:
            return jsonify({"error": "No valid image data provided"}), 400

        # Step 1: Disguise Detection
        disguise_name = detect_disguise(image_path)
        logging.info(f"Detected disguise: {disguise_name}")

        # Step 2: Patch Classification and Reconstruction
        output_image_path = os.path.join(output_clean_path, f"clean_{filename}")
        classify_and_reconstruct(image_path, output_image_path, patch_size=32)

        # Step 3: Feature Extraction
        feature_file = process_single_image(output_image_path, output_features_path)
        if feature_file:
            query_feature = np.load(feature_file)
            disguise_type = 0 if disguise_name == "clean" else 1

            # Step 4: Matching
            best_match, similarity = match_features(query_feature, disguise_type, biometric_path, non_biometric_path)

            # Return JSON response
            return jsonify({
                "disguise": disguise_name,
                "best_match": best_match,
                "similarity": f"{similarity * 100:.2f}",
                "clean_image": f"/output_clean_images/{os.path.basename(output_image_path)}"
            })
        else:
            return jsonify({"error": "Feature extraction failed"}), 500

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)'''



'''from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from PIL import Image
import logging
import base64
from io import BytesIO
import time
from disguise_detection import detect_disguise
from test_classify import classify_and_reconstruct
from feature_extraction import process_single_image
from matching import match_features

# Paths
input_image_folder = "input_images"
biometric_path = "feature_database/biometric"
non_biometric_path = "feature_database/non_biometric"
output_clean_path = "output_clean_images"
output_features_path = "output_features"

# Ensure directories exist
os.makedirs(input_image_folder, exist_ok=True)
os.makedirs(output_clean_path, exist_ok=True)
os.makedirs(output_features_path, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Flask App
app = Flask(__name__, static_url_path='', static_folder='.')

def aggregate_features(existing_feature_path, new_feature):
    """
    Aggregates existing features with a new feature using mean pooling.
    """
    if os.path.exists(existing_feature_path):
        existing_feature = np.load(existing_feature_path)
        aggregated_feature = (existing_feature + new_feature) / 2
        logging.info(f"Aggregating features at: {existing_feature_path}")
    else:
        aggregated_feature = new_feature  # Initialize with new feature if no existing feature
        logging.info(f"Initializing new aggregated feature at: {existing_feature_path}")
    return aggregated_feature

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'image' in request.files:
            file = request.files['image']
            filename = f"{int(time.time())}_{file.filename}"
            image_path = os.path.join(input_image_folder, filename)
            file.save(image_path)
        elif request.json and 'image' in request.json:
            image_data = request.json['image'].split(',')[1]
            image_data = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_data))
            filename = "webcam_image.jpg"
            image_path = os.path.join(input_image_folder, filename)
            image.save(image_path)
        else:
            return jsonify({"error": "No valid image data provided"}), 400

        disguise_name = detect_disguise(image_path)
        logging.info(f"Detected disguise: {disguise_name}")

        output_image_path = os.path.join(output_clean_path, f"clean_{filename}")
        classify_and_reconstruct(image_path, output_image_path, patch_size=32)

        feature_file = process_single_image(output_image_path, output_features_path)
        if feature_file:
            query_feature = np.load(feature_file)

            query_feature = np.squeeze(query_feature)
            if query_feature.ndim != 1:
                raise ValueError("Extracted feature vector is not 1-D")

            disguise_type = 0 if disguise_name == "clean" else 1
            best_match, similarity = match_features(query_feature, disguise_type, biometric_path, non_biometric_path)

            match_status = "matched" if similarity > 0.9 else "unmatched"
            return jsonify({
                "disguise": disguise_name,
                "best_match": best_match if match_status == "matched" else "No Match",
                "similarity": f"{similarity * 100:.2f}" if match_status == "matched" else "N/A",
                "match_status": match_status,
                "clean_image": f"/output_clean_images/{os.path.basename(output_image_path)}"
            })
        else:
            return jsonify({"error": "Feature extraction failed"}), 500

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/aggregate_feature', methods=['POST'])
def aggregate_feature():
    try:
        label = request.json.get('label')
        if not label:
            logging.error("Label missing in aggregation request")
            return jsonify({"error": "Label is required"}), 400

        person_path = os.path.join(biometric_path, label)
        aggregated_feature_path = os.path.join(person_path, f"{label}_aggregated_clean_feature.npy")
        os.makedirs(person_path, exist_ok=True)

        feature_file = os.path.join(output_features_path, f"{label}_feature.npy")
        logging.info(f"Looking for feature file: {feature_file}")

        if not os.path.exists(feature_file):
            logging.error(f"Feature file not found: {feature_file}")
            return jsonify({"error": f"Feature file {feature_file} not found"}), 400

        new_feature = np.load(feature_file)
        aggregated_feature = aggregate_features(aggregated_feature_path, new_feature)
        np.save(aggregated_feature_path, aggregated_feature)

        logging.info(f"Aggregated feature saved at {aggregated_feature_path}")
        return jsonify({"message": "Feature successfully aggregated"})
    except Exception as e:
        logging.error(f"Error in feature aggregation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/add_to_database', methods=['POST'])
def add_to_database():
    try:
        data = request.get_json()
        label = data.get('label')
        database_type = data.get('type')
        disguise_type = data.get('disguise', None)

        if not label or not database_type:
            logging.error("Label or type missing in request")
            return jsonify({"error": "Label and type are required"}), 400

        feature_file = os.path.join(output_features_path, f"{label}_feature.npy")
        if not os.path.exists(feature_file):
            logging.error(f"Feature file not found: {feature_file}")
            return jsonify({"error": f"Feature file {feature_file} not found"}), 400

        new_feature = np.load(feature_file)

        if database_type == "biometric":
            person_path = biometric_path
            aggregated_feature_path = os.path.join(person_path, f"{label}_aggregated_clean_feature.npy")
            aggregated_feature = aggregate_features(aggregated_feature_path, new_feature)
            np.save(aggregated_feature_path, aggregated_feature)
            logging.info(f"Biometric feature saved at {aggregated_feature_path}")

        elif database_type == "non_biometric":
            if not disguise_type:
                logging.error("Disguise type missing for non-biometric feature")
                return jsonify({"error": "Disguise type is required for non-biometric features"}), 400

            person_path = os.path.join(non_biometric_path, label)
            disguise_folder = os.path.join(person_path, disguise_type)
            os.makedirs(disguise_folder, exist_ok=True)
            feature_save_path = os.path.join(disguise_folder, f"{disguise_type}_feature.npy")
            np.save(feature_save_path, new_feature)
            logging.info(f"Non-biometric feature saved at {feature_save_path}")

        return jsonify({"message": "Feature successfully added to the database"})
    except Exception as e:
        logging.error(f"Error adding to database: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)'''


'''
semi working aggregate button 

from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from PIL import Image
import logging
import base64
from io import BytesIO
import time
from disguise_detection import detect_disguise
from test_classify import classify_and_reconstruct
from feature_extraction import process_single_image
from matching import match_features

# Paths
input_image_folder = "input_images"
biometric_path = "feature_database/biometric"
non_biometric_path = "feature_database/non_biometric"
output_clean_path = "output_clean_images"
output_features_path = "output_features"

# Ensure directories exist
os.makedirs(input_image_folder, exist_ok=True)
os.makedirs(output_clean_path, exist_ok=True)
os.makedirs(output_features_path, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Flask App
app = Flask(__name__, static_url_path='', static_folder='.')

def get_latest_feature():
    try:
        files = [
            os.path.join(output_features_path, f)
            for f in os.listdir(output_features_path)
            if f.endswith('.npy')  # Ensure only .npy files are considered
        ]
        if not files:
            raise FileNotFoundError("No feature files found in output_features directory.")
        latest_file = max(files, key=os.path.getctime)
        logging.info(f"Latest feature file identified: {latest_file}")
        return latest_file
    except Exception as e:
        logging.error(f"Error finding the latest feature: {e}")
        return None



def aggregate_features(existing_feature_path, new_feature):
    """
    Aggregates existing features with a new feature using mean pooling.
    """
    if os.path.exists(existing_feature_path):
        existing_feature = np.load(existing_feature_path)
        aggregated_feature = (existing_feature + new_feature) / 2
        logging.info(f"Aggregating features at: {existing_feature_path}")
    else:
        aggregated_feature = new_feature  # Initialize with new feature if no existing feature
        logging.info(f"Initializing new aggregated feature at: {existing_feature_path}")
    return aggregated_feature

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'image' in request.files:
            file = request.files['image']
            filename = f"{int(time.time())}_{file.filename}"
            image_path = os.path.join(input_image_folder, filename)
            file.save(image_path)
        elif request.json and 'image' in request.json:
            image_data = request.json['image'].split(',')[1]
            image_data = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_data))
            filename = "webcam_image.jpg"
            image_path = os.path.join(input_image_folder, filename)
            image.save(image_path)
        else:
            return jsonify({"error": "No valid image data provided"}), 400

        disguise_name = detect_disguise(image_path)
        logging.info(f"Detected disguise: {disguise_name}")

        output_image_path = os.path.join(output_clean_path, f"clean_{filename}")
        classify_and_reconstruct(image_path, output_image_path, patch_size=32)

        feature_file = process_single_image(output_image_path, output_features_path)
        if feature_file:
            query_feature = np.load(feature_file)

            query_feature = np.squeeze(query_feature)
            if query_feature.ndim != 1:
                raise ValueError("Extracted feature vector is not 1-D")

            disguise_type = 0 if disguise_name == "clean" else 1
            best_match, similarity = match_features(query_feature, disguise_type, biometric_path, non_biometric_path)

            match_status = "matched" if similarity > 0.9 else "unmatched"
            return jsonify({
                "disguise": disguise_name,
                "best_match": best_match if match_status == "matched" else "No Match",
                "similarity": f"{similarity * 100:.2f}" if match_status == "matched" else "N/A",
                "match_status": match_status,
                "clean_image": f"/output_clean_images/{os.path.basename(output_image_path)}"
            })
        else:
            return jsonify({"error": "Feature extraction failed"}), 500

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/aggregate_feature', methods=['POST'])
def aggregate_feature():
    try:
        data = request.get_json()
        matched_person = data.get('matched_person')
        disguise = data.get('disguise')
        if not matched_person or not disguise:
            return jsonify({"error": "Matched person and disguise type are required for aggregation"}), 400

        latest_feature_file = get_latest_feature()
        if not latest_feature_file:
            return jsonify({"error": "No latest feature file found"}), 400

        new_feature = np.load(latest_feature_file)
        person_path = os.path.join(non_biometric_path, matched_person)
        disguise_feature_path = os.path.join(person_path, f"{disguise}_feature.npy")
        os.makedirs(person_path, exist_ok=True)

        aggregated_feature = aggregate_features(disguise_feature_path, new_feature)
        np.save(disguise_feature_path, aggregated_feature)
        logging.info(f"Aggregated feature saved at {disguise_feature_path}")

        return jsonify({"message": "Feature successfully aggregated"})
    except Exception as e:
        logging.error(f"Error during feature aggregation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/add_to_database', methods=['POST'])
def add_to_database():
    try:
        data = request.get_json()
        label = data.get('label')
        database_type = data.get('type')
        disguise_type = data.get('disguise')

        if not label or not database_type:
            return jsonify({"error": "Label and type are required"}), 400

        latest_feature_file = get_latest_feature()
        if not latest_feature_file:
            return jsonify({"error": "No latest feature file found"}), 400

        new_feature = np.load(latest_feature_file)

        if database_type == "biometric":
            person_path = biometric_path
            aggregated_feature_path = os.path.join(person_path, f"{label}_aggregated_clean_feature.npy")
            aggregated_feature = aggregate_features(aggregated_feature_path, new_feature)
            np.save(aggregated_feature_path, aggregated_feature)
            logging.info(f"Biometric feature saved at {aggregated_feature_path}")

        elif database_type == "non_biometric":
            if not disguise_type:
                return jsonify({"error": "Disguise type is required for non-biometric features"}), 400
            person_path = os.path.join(non_biometric_path, label)
            os.makedirs(person_path, exist_ok=True)
            feature_save_path = os.path.join(person_path, f"{disguise_type}_feature.npy")
            np.save(feature_save_path, new_feature)
            logging.info(f"Non-biometric feature saved at {feature_save_path}")

        return jsonify({"message": "Feature successfully added to the database"}), 200
    except Exception as e:
        logging.error(f"Error adding feature to database: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)'''



'''working code of app.py fully functional but older code'''

from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from PIL import Image
import logging
import base64
from io import BytesIO
import time
from disguise_detection import detect_disguise
from test_classify import classify_and_reconstruct
from feature_extraction import process_single_image
from matching import match_features

# Paths
input_image_folder = "input_images"
biometric_path = "feature_database/biometric"
non_biometric_path = "feature_database/non_biometric"
output_clean_path = "output_clean_images"
output_features_path = "output_features"

# Ensure directories exist
os.makedirs(input_image_folder, exist_ok=True)
os.makedirs(output_clean_path, exist_ok=True)
os.makedirs(output_features_path, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__, static_url_path='', static_folder='.')

#########################
# AUTO-INCREMENT GLOBAL
#########################
person_count = 1  # Will be set via scan_existing_labels()

def scan_existing_labels():
    """
    Scans the biometric and non_biometric directories to find existing person labels,
    like person64. Determines the maximum label and sets person_count accordingly.
    """
    global person_count
    max_label_num = 0

    # Check biometric folder for files like person64_aggregated_clean_feature.npy
    if os.path.exists(biometric_path):
        for file_name in os.listdir(biometric_path):
            if file_name.startswith("person") and file_name.endswith("_aggregated_clean_feature.npy"):
                # Extract the integer, e.g., '64' from 'person64_aggregated_clean_feature.npy'
                label_str = file_name.split("_aggregated_clean_feature.npy")[0].replace("person", "")
                try:
                    label_num = int(label_str)
                    max_label_num = max(max_label_num, label_num)
                except ValueError:
                    pass

    # Check non_biometric folder for subfolders named personX
    if os.path.exists(non_biometric_path):
        for folder_name in os.listdir(non_biometric_path):
            if folder_name.startswith("person"):
                label_str = folder_name.replace("person", "")
                try:
                    label_num = int(label_str)
                    max_label_num = max(max_label_num, label_num)
                except ValueError:
                    pass

    person_count = max_label_num + 1
    logging.info(f"Initialized person_count to {person_count} from existing labels.")

#####################
# HELPER FUNCTIONS
#####################

def get_latest_feature():
    """
    Returns the most recently modified .npy file from output_features,
    or None if none exist.
    """
    try:
        files = [
            os.path.join(output_features_path, f)
            for f in os.listdir(output_features_path)
            if f.endswith('.npy')
        ]
        if not files:
            raise FileNotFoundError("No feature files found in output_features directory.")
        latest_file = max(files, key=os.path.getctime)
        logging.info(f"Latest feature file identified: {latest_file}")
        return latest_file
    except Exception as e:
        logging.error(f"Error finding the latest feature: {e}")
        return None

def aggregate_features(existing_feature_path, new_feature):
    """
    Aggregates existing features with a new feature using mean pooling.
    """
    if os.path.exists(existing_feature_path):
        existing_feature = np.load(existing_feature_path)
        aggregated_feature = (existing_feature + new_feature) / 2
        logging.info(f"Aggregating features at: {existing_feature_path}")
    else:
        aggregated_feature = new_feature
        logging.info(f"Initializing new aggregated feature at: {existing_feature_path}")
    return aggregated_feature

#####################
# ROUTES
#####################

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """
    Upload route handles both file-based and webcam images.
    If matched, returns matched data. If unmatched, returns 'unmatched' status
    without assigning a new label or storing the feature in the final database path.
    """
    try:
        # 1. Get image from request
        if 'image' in request.files:
            file = request.files['image']
            filename = f"{int(time.time())}_{file.filename}"
            image_path = os.path.join(input_image_folder, filename)
            file.save(image_path)
        elif request.json and 'image' in request.json:
            image_data = request.json['image'].split(',')[1]
            image_data = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_data))
            filename = "webcam_image.jpg"
            image_path = os.path.join(input_image_folder, filename)
            image.save(image_path)
        else:
            return jsonify({"error": "No valid image data provided"}), 400

        # 2. Disguise Detection
        disguise_name = detect_disguise(image_path)
        logging.info(f"Detected disguise: {disguise_name}")

        # 3. Reconstruction
        output_image_path = os.path.join(output_clean_path, f"clean_{filename}")
        classify_and_reconstruct(image_path, output_image_path, patch_size=32)

        # 4. Feature Extraction
        feature_file = process_single_image(output_image_path, output_features_path)
        if not feature_file:
            return jsonify({"error": "Feature extraction failed"}), 500

        query_feature = np.load(feature_file)
        query_feature = np.squeeze(query_feature)
        if query_feature.ndim != 1:
            raise ValueError("Extracted feature vector is not 1-D")

        # 5. Matching
        disguise_type = 0 if disguise_name == "clean" else 1
        best_match, similarity = match_features(query_feature, disguise_type, biometric_path, non_biometric_path)
        match_status = "matched" if similarity > 0.9 else "unmatched"

        if match_status == "matched":
            return jsonify({
                "disguise": disguise_name,
                "best_match": best_match,
                "similarity": f"{similarity * 100:.2f}",
                "match_status": match_status,
                "clean_image": f"/output_clean_images/{os.path.basename(output_image_path)}"
            })
        else:
            # Unmatched -> do NOT assign or store in DB yet, just return unmatched
            return jsonify({
                "disguise": disguise_name,
                "best_match": "No Match",
                "similarity": "N/A",
                "match_status": "unmatched",
                "clean_image": f"/output_clean_images/{os.path.basename(output_image_path)}"
            })

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/aggregate_feature', methods=['POST'])
def aggregate_feature():
    """
    Aggregates the latest feature with an existing matched person & disguise
    in non_biometric (or you'd adapt for clean).
    """
    try:
        data = request.get_json()
        matched_person = data.get('matched_person')
        disguise = data.get('disguise')

        if not matched_person or not disguise:
            return jsonify({"error": "Matched person and disguise type are required"}), 400

        latest_feature_file = get_latest_feature()
        if not latest_feature_file:
            return jsonify({"error": "No latest feature file found"}), 400

        new_feature = np.load(latest_feature_file)

        # If you'd like to handle 'clean' aggregation in biometric,
        # you'd do something like:
        # if disguise == "clean": store in biometric
        # else: store in non_biometric
        person_path = os.path.join(non_biometric_path, matched_person)
        os.makedirs(person_path, exist_ok=True)
        disguise_feature_path = os.path.join(person_path, f"{disguise}_feature.npy")

        aggregated = aggregate_features(disguise_feature_path, new_feature)
        np.save(disguise_feature_path, aggregated)
        logging.info(f"Aggregated feature saved at {disguise_feature_path}")

        return jsonify({"message": "Feature successfully aggregated"}), 200

    except Exception as e:
        logging.error(f"Error during feature aggregation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/add_to_database', methods=['POST'])
def add_to_database():
    """
    User triggers this if the image was unmatched. We auto-assign a new label
    and store the latest feature in either biometric or non_biometric.
    """
    global person_count

    try:
        data = request.get_json()
        disguise = data.get('disguise')
        if not disguise:
            return jsonify({"error": "Disguise type is required"}), 400

        latest_feature_file = get_latest_feature()
        if not latest_feature_file:
            return jsonify({"error": "No latest feature file found"}), 400

        new_feature = np.load(latest_feature_file)
        # Auto-increment a new label
        new_label = f"person{person_count}"
        person_count += 1

        if disguise.lower() == "clean":
            # Save to biometric
            aggregated_path = os.path.join(biometric_path, f"{new_label}_aggregated_clean_feature.npy")
            agg = aggregate_features(aggregated_path, new_feature)
            np.save(aggregated_path, agg)
            logging.info(f"Created new label {new_label} (biometric) at {aggregated_path}")
        else:
            # Save to non-biometric
            person_path = os.path.join(non_biometric_path, new_label)
            os.makedirs(person_path, exist_ok=True)
            feature_path = os.path.join(person_path, f"{disguise}_feature.npy")
            agg = aggregate_features(feature_path, new_feature)
            np.save(feature_path, agg)
            logging.info(f"Created new label {new_label} (non-biometric) at {feature_path}")

        return jsonify({
            "message": f"Successfully added new label {new_label}",
            "label": new_label
        }), 200

    except Exception as e:
        logging.error(f"Error adding feature to database: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    def scan_existing_labels():
        global person_count
        max_label_num = 0

        # Check biometric folder for personX_aggregated_clean_feature
        if os.path.exists(biometric_path):
            for f_name in os.listdir(biometric_path):
                if f_name.startswith("person") and f_name.endswith("_aggregated_clean_feature.npy"):
                    label_str = f_name.split("_aggregated_clean_feature.npy")[0].replace("person", "")
                    try:
                        lbl_num = int(label_str)
                        max_label_num = max(max_label_num, lbl_num)
                    except ValueError:
                        pass

        # Check non_biometric folder
        if os.path.exists(non_biometric_path):
            for folder_name in os.listdir(non_biometric_path):
                if folder_name.startswith("person"):
                    label_str = folder_name.replace("person", "")
                    try:
                        lbl_num = int(label_str)
                        max_label_num = max(max_label_num, lbl_num)
                    except ValueError:
                        pass

        person_count = max_label_num + 1
        logging.info(f"Initialized person_count to {person_count} from existing labels.")

    scan_existing_labels()
    app.run(debug=True)




















