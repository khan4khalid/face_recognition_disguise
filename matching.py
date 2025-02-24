import os
import numpy as np
from scipy.spatial.distance import cosine

def match_features(query_feature, disguise_type, biometric_db, non_biometric_db):
    """
    Match query features against the appropriate database.
    Args:
        query_feature (np.ndarray): Feature vector of the query image.
        disguise_type (int): Detected disguise type (0 for no disguise).
        biometric_db (str): Path to the biometric database.
        non_biometric_db (str): Path to the non-biometric database.
    Returns:
        str: Best match person ID.
        float: Similarity score of the best match.
    """
    database_path = biometric_db if disguise_type == 0 else non_biometric_db
    best_match = None
    highest_similarity = -1

    for person_folder in os.listdir(database_path):
        person_path = os.path.join(database_path, person_folder)
        if not os.path.isdir(person_path):
            continue

        for feature_file in os.listdir(person_path):
            feature_path = os.path.join(person_path, feature_file)
            database_feature = np.load(feature_path)
            similarity = 1 - cosine(query_feature, database_feature)

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = person_folder

    return best_match, highest_similarity



'''semi working not fully working , it is showing is na in web app 
import os
import numpy as np
from scipy.spatial.distance import cosine

def match_features(query_feature, disguise_type, biometric_db, non_biometric_db):
    """
    Search the relevant database (biometric if disguise_type == 0,
    non_biometric if disguise_type != 0) by recursively walking every folder
    and subfolder for .npy embeddings. Picks the top-level 'person' folder as 
    best match if it yields the highest similarity.

    Args:
        query_feature (np.ndarray): Feature vector from the pipeline (shape e.g. (512,)).
        disguise_type (int): 0 => 'clean' => use biometric_db. 
                             anything else => use non_biometric_db.
        biometric_db (str): path to the 'biometric' folder.
        non_biometric_db (str): path to the 'non_biometric' folder.

    Returns:
        best_match (str): Name of the person folder with highest similarity
        highest_similarity (float): The best similarity [0..1] found
    """

    # Pick which database path to search
    db_path = biometric_db if disguise_type == 0 else non_biometric_db

    best_match = None
    highest_similarity = -1.0

    # 1) Iterate each top-level person folder
    for person_folder in os.listdir(db_path):
        person_path = os.path.join(db_path, person_folder)
        if not os.path.isdir(person_path):
            continue

        # We'll keep track of the best similarity for THIS person (some .npy inside subfolders)
        best_for_this_person = -1.0

        # 2) Recursively walk all subfolders to find .npy
        for root, dirs, files in os.walk(person_path):
            for file in files:
                if not file.endswith(".npy"):
                    continue
                feature_path = os.path.join(root, file)

                # Load the database embedding
                database_feature = np.load(feature_path)

                # Cosine distance => similarity
                sim = 1 - cosine(query_feature, database_feature)
                if sim > best_for_this_person:
                    best_for_this_person = sim

        # 3) If this person's best is globally the best, update
        if best_for_this_person > highest_similarity:
            highest_similarity = best_for_this_person
            best_match = person_folder

    return best_match, highest_similarity

'''

'''arcface matching function
import os
import numpy as np
from scipy.spatial.distance import cosine

def match_features(query_feature, disguise_name, biometric_db, non_biometric_db):
    """
    Match query features based on the disguise name:
    - If disguise_name.lower() == "clean", we search in biometric_db/personX/* for .npy
    - Else we search in non_biometric_db/personX/<disguise_name>/* for .npy
    Then find the person with highest similarity.

    Args:
        query_feature (np.ndarray): (512,) embedding from ArcFace
        disguise_name (str): "clean", "Mask", "Beard", "Cap", etc.
        biometric_db (str): path to the 'feature_database/biometric'
        non_biometric_db (str): path to the 'feature_database/non_biometric'

    Returns:
        (best_person, highest_similarity)
        best_person: name of top-level folder (e.g. "person1")
        highest_similarity: float in [0..1] if vectors are L2 normalized
    """
    # Decide if we search in biometric or non_biometric
    if disguise_name.lower() == "clean":
        db_path = biometric_db  # e.g. feature_database/biometric
        is_clean = True
    else:
        db_path = non_biometric_db  # e.g. feature_database/non_biometric
        is_clean = False

    best_person = None
    highest_similarity = -1.0

    # 1) Iterate each top-level "personX" folder
    for person_folder in os.listdir(db_path):
        person_path = os.path.join(db_path, person_folder)
        if not os.path.isdir(person_path):
            continue

        best_for_this_person = -1.0

        if is_clean:
            # For clean => we expect .npy directly under personX/ 
            # e.g. personX/clean_feature.npy or multiple .npy
            if not os.path.isdir(person_path):
                continue
            # search all .npy in person_path
            for file in os.listdir(person_path):
                if file.endswith(".npy"):
                    emb_path = os.path.join(person_path, file)
                    db_emb = np.load(emb_path)
                    sim = 1 - cosine(query_feature, db_emb)
                    if sim > best_for_this_person:
                        best_for_this_person = sim
        else:
            # For disguised => look for subfolder personX/disguise_name
            subfolder_path = os.path.join(person_path, disguise_name)
            if not os.path.isdir(subfolder_path):
                # Person doesn't have that disguise subfolder => skip
                continue
            for file in os.listdir(subfolder_path):
                if file.endswith(".npy"):
                    emb_path = os.path.join(subfolder_path, file)
                    db_emb = np.load(emb_path)
                    sim = 1 - cosine(query_feature, db_emb)
                    if sim > best_for_this_person:
                        best_for_this_person = sim

        # if this person's best is better than the global best
        if best_for_this_person > highest_similarity:
            highest_similarity = best_for_this_person
            best_person = person_folder

    return best_person, highest_similarity'''














