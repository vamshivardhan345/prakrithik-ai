import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from utils.pesticide_recommendation import recommender

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_plant_model.keras")
CLASSES_PATH = os.path.join(BASE_DIR, "model", "classes.json")

# Minimum confidence required to accept a prediction
CONFIDENCE_THRESHOLD = 0.35

class PlantDiseasePredictor:
    """
    Production-grade inference engine for plant disease classification using MobileNetV2.
    Integrates confidence filtering, out-of-distribution leaf detection,
    and automated organic treatment mapping.
    """
    def __init__(self, model_path: str = MODEL_PATH, classes_path: str = CLASSES_PATH):
        self.model_path = model_path
        self.classes_path = classes_path
        self.model = None
        self.classes = []
        self._load_resources()

    def _load_resources(self):
        # Load classes
        if os.path.exists(self.classes_path):
            with open(self.classes_path, "r") as f:
                self.classes = json.load(f)
            print(f"Loaded {len(self.classes)} classes from {self.classes_path}")
        else:
            print(f"Warning: Classes file not found at {self.classes_path}")

        # Load model
        if os.path.exists(self.model_path):
            print(f"Loading trained MobileNetV2 model from {self.model_path}...")
            self.model = tf.keras.models.load_model(self.model_path)
            print("Model loaded successfully into memory.")
        else:
            print(f"Warning: Model file not yet available at {self.model_path}")

    def reload_if_needed(self):
        """Allows hot-reloading once training finishes if app started earlier."""
        if self.model is None and os.path.exists(self.model_path):
            self._load_resources()

    @staticmethod
    def _is_probable_leaf_image(img: Image.Image) -> bool:
        """
        Heuristic image check to reject non-image inputs, solid blank frames,
        or completely monochrome non-plant files.
        """
        # Convert to RGB and check variance
        stat_img = img.resize((32, 32))
        arr = np.array(stat_img, dtype=np.float32)
        std_per_channel = np.std(arr, axis=(0, 1))
        # If std is near zero, image is blank/solid single color
        if np.mean(std_per_channel) < 8.0:
            return False
        return True

    def preprocess_image(self, img_path: str):
        """
        Validates, resizes to (224, 224), and applies MobileNetV2 preprocessing.
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at path: {img_path}")

        try:
            img = Image.open(img_path)
            img.verify()  # Integrity check
        except Exception as e:
            raise ValueError(f"Corrupted or invalid image file: {str(e)}")

        # Reopen for actual processing (verify closes/invalidates the handle)
        img = Image.open(img_path).convert("RGB")
        
        if not self._is_probable_leaf_image(img):
            raise ValueError("The uploaded image does not appear to contain a recognizable plant leaf or is completely blank. Please upload a clear photo of a plant leaf.")

        # Resize to standard MobileNetV2 input
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        
        # The model architecture includes MobileNetV2 preprocess_input inside its graph,
        # so it takes raw [0, 255] float32 image tensors directly.
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    @staticmethod
    def clean_name(raw: str) -> str:
        if "___" in raw:
            parts = raw.split("___")
            crop = parts[0].replace("_", " ").strip()
            disease = parts[1].replace("_", " ").strip()
            return f"{crop} - {disease}"
        return raw.replace("_", " ").strip()

    def predict(self, img_path: str):
        """
        Runs model prediction and retrieves organic pesticide recommendation.
        """
        self.reload_if_needed()
        if self.model is None:
            return {
                "success": False,
                "error": "Model is not loaded or training is still finalizing. Please try again in a moment."
            }

        try:
            # 1. Preprocess
            input_tensor = self.preprocess_image(img_path)

            # 2. Model Inference
            raw_preds = self.model.predict(input_tensor, verbose=0)[0]
            raw_preds = np.nan_to_num(raw_preds)

            confidence = float(np.max(raw_preds))
            pred_idx = int(np.argmax(raw_preds))
            predicted_raw_class = self.classes[pred_idx]

            # 3. Confidence Threshold Check
            if confidence < CONFIDENCE_THRESHOLD:
                return {
                    "success": False,
                    "uncertain": True,
                    "confidence": confidence,
                    "confidence_percent": f"{confidence * 100:.1f}%",
                    "error": (
                        f"Uncertain diagnosis ({confidence * 100:.1f}% confidence, below {CONFIDENCE_THRESHOLD * 100:.0f}% threshold). "
                        "The uploaded image could not be reliably classified. Please ensure the image is a clear, well-lit close-up of a single plant leaf."
                    )
                }

            # 4. Top Predictions Ranking
            top_k_indices = raw_preds.argsort()[-3:][::-1]
            top_predictions = []
            for idx in top_k_indices:
                top_predictions.append({
                    "raw_name": self.classes[idx],
                    "name": self.clean_name(self.classes[idx]),
                    "prob": float(raw_preds[idx]),
                    "percent": f"{float(raw_preds[idx]) * 100:.1f}%"
                })

            # 5. Extract Crop & Disease
            crop, disease = recommender.parse_class_name(predicted_raw_class)

            # 6. Retrieve Organic Pesticide Recommendation from CSV
            rec = recommender.get_recommendation(predicted_raw_class)

            return {
                "success": True,
                "raw_class": predicted_raw_class,
                "crop": crop,
                "disease": disease,
                "display_name": self.clean_name(predicted_raw_class),
                "confidence": confidence,
                "confidence_percent": f"{confidence * 100:.1f}%",
                "top_predictions": top_predictions,
                "recommendation": rec
            }

        except ValueError as ve:
            return {"success": False, "error": str(ve)}
        except Exception as e:
            return {"success": False, "error": f"Internal prediction error: {str(e)}"}

# Global singleton
predictor = PlantDiseasePredictor()
