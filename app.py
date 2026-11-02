import os
import logging
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Custom inference and recommendation modules
from utils.disease_prediction import predictor
from utils.pesticide_recommendation import recommender

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "prakritik_agricultural_ai_key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image file provided in request."}), 400
        
    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({"success": False, "error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported file format. Please upload a valid image file (JPG, PNG, JPEG, WEBP)."
        }), 400

    filepath = None
    try:
        # Secure filename and save
        safe_name = secure_filename(file.filename)
        # Avoid collisions with timestamp
        unique_name = f"upload_{os.urandom(4).hex()}_{safe_name}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(filepath)

        # Run AI Disease Diagnosis + Organic Recommendation Pipeline
        result = predictor.predict(filepath)

        # Attach image web path for frontend rendering
        if result.get("success"):
            result["image_url"] = f"/static/uploads/{unique_name}"

        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Global Predict Route Error: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Server processing error: {str(e)}"}), 500

@app.route("/api/recommendations", methods=["GET"])
def get_recommendations():
    """
    Search endpoint for the Organic Pesticide Recommendations browser.
    Supports filtering by crop and disease/pest.
    """
    try:
        crop = request.args.get("crop", "").strip()
        disease = request.args.get("disease", "").strip()
        results = recommender.search_catalog(crop_query=crop, disease_query=disease)
        return jsonify({
            "success": True,
            "count": len(results),
            "recommendations": results
        })
    except Exception as e:
        logger.error(f"Recommendation API Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/crops", methods=["GET"])
def get_crops():
    """Returns unique list of crops available in the recommendation database."""
    try:
        crops = sorted(recommender.df['crop'].dropna().unique().tolist())
        return jsonify({"success": True, "crops": crops})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
