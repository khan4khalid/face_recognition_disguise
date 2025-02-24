import os
import numpy as np
import logging

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

def save_feature_to_database(label, database_type, feature, biometric_path, non_biometric_path, disguise_type=None):
    """
    Saves a feature to the appropriate database location.
    """
    try:
        if database_type == "biometric":
            person_path = biometric_path
            aggregated_feature_path = os.path.join(person_path, f"{label}_aggregated_clean_feature.npy")
            aggregated_feature = aggregate_features(aggregated_feature_path, feature)
            np.save(aggregated_feature_path, aggregated_feature)
            return {"message": f"Biometric feature saved at {aggregated_feature_path}"}

        elif database_type == "non_biometric":
            if not disguise_type:
                raise ValueError("Disguise type is required for non-biometric features")

            person_path = os.path.join(non_biometric_path, label)
            disguise_folder = os.path.join(person_path, disguise_type)
            os.makedirs(disguise_folder, exist_ok=True)
            feature_save_path = os.path.join(disguise_folder, f"{disguise_type}_feature.npy")
            np.save(feature_save_path, feature)
            return {"message": f"Non-biometric feature saved at {feature_save_path}"}

        else:
            raise ValueError("Invalid database type")

    except Exception as e:
        logging.error(f"Error saving feature to database: {e}")
        return {"error": str(e)}
