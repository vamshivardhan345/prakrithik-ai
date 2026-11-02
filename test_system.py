import os
import glob
import numpy as np
from PIL import Image
from utils.disease_prediction import predictor
from utils.pesticide_recommendation import recommender

print("=" * 60)
print("COMPREHENSIVE SYSTEM VERIFICATION TEST")
print("=" * 60)

# Test 1: Real test images
test_classes = [
    "Apple___Apple_scab",
    "Corn_(maize)___Common_rust_",
    "Tomato___healthy",
    "Tomato___Early_blight"
]

print("\n--- TEST 1: Model Prediction & Organic Recommendation on Real Test Images ---")
for cls in test_classes:
    cls_folder = os.path.join("dataset", "test", cls)
    images = glob.glob(os.path.join(cls_folder, "*.jpg")) + glob.glob(os.path.join(cls_folder, "*.JPG"))
    if not images:
        print(f"No test images found for {cls}")
        continue
    
    # Test first 2 images
    for test_img in images[:2]:
        res = predictor.predict(test_img)
        print(f"\nTarget Class:       {cls}")
        print(f"Image:              {os.path.basename(test_img)}")
        print(f"Success:            {res.get('success')}")
        if res.get('success'):
            print(f"Predicted Disease:  {res.get('display_name')}")
            print(f"Confidence:         {res.get('confidence_percent')}")
            rec = res.get('recommendation', {})
            print(f"Organic Remedy:     {rec.get('organic_recommendation')}")
            print(f"Active Ingredient:  {rec.get('active_ingredients')}")
            print(f"Dosage:             {rec.get('dosage')}")
            print(f"Status Match:       {rec.get('matched')}")
        else:
            print(f"Notice:             {res.get('error')}")

# Test 2: Unmatched disease fallback
print("\n--- TEST 2: Fallback When Recommendation Is Unavailable ---")
fallback_rec = recommender.get_recommendation("Blueberry___unknown_viral_disease")
print(f"Organic Remedy:     {fallback_rec.get('organic_recommendation')}")
print(f"Matched Flag:       {fallback_rec.get('matched')}")
assert "No verified organic recommendation available for this disease." in fallback_rec.get('organic_recommendation')
print("Fallback verified!")

# Test 3: Blank / Invalid non-leaf image rejection
print("\n--- TEST 3: Invalid / Blank Image Rejection ---")
blank_path = "results/test_blank.jpg"
blank_img = Image.new("RGB", (224, 224), color=(255, 255, 255))
blank_img.save(blank_path)

res_blank = predictor.predict(blank_path)
print(f"Success: {res_blank.get('success')}")
print(f"Error Msg: {res_blank.get('error')}")
assert res_blank["success"] is False
print("Blank image rejection verified!")
if os.path.exists(blank_path):
    os.remove(blank_path)

# Test 4: Recommendation Search API
print("\n--- TEST 4: Recommendation Catalog Search ---")
tomato_recs = recommender.search_catalog(crop_query="Tomato", disease_query="blight")
print(f"Found {len(tomato_recs)} recommendations for Tomato + blight:")
for r in tomato_recs:
    print(f"  - {r['crop']} | {r['disease_or_pest']}: {r['organic_recommendation'][:45]}...")
assert len(tomato_recs) >= 2
print("Catalog search verified!")

print("\nALL SYSTEM TESTS PASSED SUCCESSFULLY!")
