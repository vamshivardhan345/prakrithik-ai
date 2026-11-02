import os
import re
import pandas as pd
from typing import Dict, Any, List, Optional

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "organic_pesticide_recommendations.csv")

class PesticideRecommender:
    """
    Organic Pesticide Recommendation engine powered by the verified CSV database.
    Integrates crop and disease matching, prevention guidelines, dosage,
    and safety precautions.
    """
    def __init__(self, csv_path: str = CSV_PATH):
        self.csv_path = csv_path
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Recommendation CSV not found at: {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        # Normalize column names for safe access
        self.df.columns = [c.strip().lower() for c in self.df.columns]
        print(f"Loaded {len(self.df)} organic pesticide recommendations from {self.csv_path}")

    @staticmethod
    def _clean_str(text: str) -> str:
        if not isinstance(text, str):
            return ""
        # Remove special characters, extra spaces, and lowercase
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', text).lower()
        return " ".join(cleaned.split())

    @staticmethod
    def parse_class_name(raw_class: str):
        """
        Extracts crop name and disease name from PlantVillage class format
        (e.g., 'Tomato___Early_blight' or 'Corn_(maize)___Common_rust_').
        """
        if "___" in raw_class:
            parts = raw_class.split("___")
            crop = parts[0].replace("_", " ").strip()
            disease = parts[1].replace("_", " ").strip()
        else:
            parts = raw_class.split(" ")
            crop = parts[0]
            disease = " ".join(parts[1:]) if len(parts) > 1 else raw_class

        # Clean crop aliases
        crop_clean = crop.lower()
        if "corn" in crop_clean or "maize" in crop_clean:
            crop = "Corn"
        elif "pepper" in crop_clean or "chilli" in crop_clean or "chili" in crop_clean:
            crop = "Chilli"
        elif "apple" in crop_clean:
            crop = "Apple"
        elif "grape" in crop_clean:
            crop = "Grape"
        elif "tomato" in crop_clean:
            crop = "Tomato"
        elif "potato" in crop_clean:
            crop = "Potato"
        elif "rice" in crop_clean:
            crop = "Rice"
        elif "brinjal" in crop_clean or "eggplant" in crop_clean:
            crop = "Brinjal"

        return crop, disease

    def get_recommendation(self, raw_class_name: str) -> Dict[str, Any]:
        """
        Given a PlantVillage class or crop+disease, look up verified organic treatment.
        """
        crop, disease = self.parse_class_name(raw_class_name)
        disease_clean = self._clean_str(disease)
        crop_clean = self._clean_str(crop)

        # 1. Handle Healthy Plants
        if "healthy" in disease_clean:
            return {
                "matched": True,
                "crop": crop,
                "disease": "Healthy Plant (No Disease Detected)",
                "organic_recommendation": "No pesticide required. Your plant appears vigorous and disease-free.",
                "active_ingredients": "Not applicable - Foliage is healthy",
                "application_guidance": "Continue standard preventive cultural care: maintain balanced soil moisture, proper aeration, and provide well-rotted organic compost.",
                "dosage": "Not applicable",
                "frequency": "Routine seasonal plant care",
                "integrated_prevention": "Rotate crops every season; maintain adequate plant spacing for airflow; scout foliage weekly for early signs of pests.",
                "safety_precautions": "Follow safe organic farming practices. Protect beneficial pollinators like bees and ladybugs.",
                "validation_status": "Healthy baseline verified",
                "reference_to_verify": "ICAR; State Agricultural University Extension Guidelines",
                "disclaimer": "Recommendations must be checked against local agricultural guidance and the product label before application."
            }

        # 2. Search for matching row in CSV
        # Strategy A: Match specific crop AND disease
        matched_row = None
        
        # Check rows for this specific crop
        crop_rows = self.df[self.df['crop'].apply(lambda x: self._clean_str(str(x)) == crop_clean or crop_clean in self._clean_str(str(x)))]
        
        for _, row in crop_rows.iterrows():
            row_disease = self._clean_str(str(row.get('disease_or_pest', '')))
            # Check overlap between disease strings
            words_a = set(disease_clean.split())
            words_b = set(row_disease.split())
            common = words_a.intersection(words_b)
            # Exclude trivial stop words
            common = {w for w in common if w not in {'leaf', 'spot', 'rot', 'blight', 'virus', 'and', 'the', 'or'}}
            
            # Check if key disease terms match
            if row_disease in disease_clean or disease_clean in row_disease or len(common) > 0:
                matched_row = row
                break

        # Strategy B: If no match yet, check "Multiple crops" in CSV (e.g. Spider mites, caterpillars)
        if matched_row is None:
            multi_rows = self.df[self.df['crop'].str.lower().str.contains("multiple", na=False)]
            for _, row in multi_rows.iterrows():
                row_disease = self._clean_str(str(row.get('disease_or_pest', '')))
                if row_disease and (row_disease in disease_clean or any(w in disease_clean for w in row_disease.split())):
                    matched_row = row
                    break

        # Strategy C: Check across all rows by disease name if crop is an alias or variant
        if matched_row is None:
            for _, row in self.df.iterrows():
                row_disease = self._clean_str(str(row.get('disease_or_pest', '')))
                if disease_clean and row_disease and (disease_clean == row_disease or (len(disease_clean) > 5 and disease_clean in row_disease)):
                    matched_row = row
                    break

        # 3. Format Response
        if matched_row is not None:
            def safe_field(key: str, default: str = "As directed on product label") -> str:
                val = matched_row.get(key)
                if pd.isna(val) or str(val).strip() == "" or str(val).strip().lower() == "nan":
                    return default
                return str(val).strip()

            return {
                "matched": True,
                "crop": safe_field("crop", crop),
                "disease": safe_field("disease_or_pest", disease),
                "issue_type": safe_field("issue_type", "Disease"),
                "causal_organism": safe_field("causal_organism_or_pest_family", "Pathogen / Pest"),
                "organic_recommendation": safe_field("organic_recommendation"),
                "active_ingredients": safe_field("active_ingredient_or_biological_agent", "Biological / botanical agent"),
                "application_guidance": safe_field("application_guidance"),
                "dosage": safe_field("dosage", "Label-specific; no generic dose"),
                "frequency": safe_field("frequency", "Only as directed on product label"),
                "integrated_prevention": safe_field("integrated_prevention"),
                "safety_precautions": safe_field("safety_precautions"),
                "validation_status": safe_field("validation_status", "Candidate recommendation - verify locally"),
                "reference_to_verify": safe_field("reference_to_verify", "ICAR / CIB&RC / State Agriculture University"),
                "disclaimer": "CRITICAL NOTICE: Recommendations must be checked against local agricultural guidance and the product label before application. Do not use unapproved chemical substitutes or unverified generic dosages."
            }

        # 4. Fallback when no verified recommendation is in CSV
        return {
            "matched": False,
            "crop": crop,
            "disease": disease,
            "organic_recommendation": "No verified organic recommendation available for this disease.",
            "active_ingredients": "Not applicable",
            "application_guidance": "Please consult your local Krishi Vigyan Kendra (KVK), ICAR center, or certified agricultural university extension officer for verified organic interventions.",
            "dosage": "Label-specific; do not use unverified dosages",
            "frequency": "Not applicable",
            "integrated_prevention": "Remove and safely dispose of infected crop debris; improve canopy aeration; avoid overhead sprinkler irrigation; practice 2-3 year crop rotation with non-host species.",
            "safety_precautions": "Avoid applying unverified home remedies or unregistered concoctions which may burn crop foliage.",
            "validation_status": "No database record match",
            "reference_to_verify": "Consult your local Agricultural Extension Office / KVK",
            "disclaimer": "CRITICAL NOTICE: Recommendations must be checked against local agricultural guidance and the product label before application."
        }

    def search_catalog(self, crop_query: Optional[str] = None, disease_query: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search catalog for the Recommendation Browser page.
        """
        filtered = self.df.copy()
        if crop_query and crop_query.strip():
            c_clean = self._clean_str(crop_query)
            filtered = filtered[filtered['crop'].apply(lambda x: c_clean in self._clean_str(str(x)))]
            
        if disease_query and disease_query.strip():
            d_clean = self._clean_str(disease_query)
            filtered = filtered[filtered['disease_or_pest'].apply(lambda x: d_clean in self._clean_str(str(x)))]

        results = []
        for _, row in filtered.iterrows():
            results.append({
                "crop": str(row.get('crop', '')),
                "disease_or_pest": str(row.get('disease_or_pest', '')),
                "issue_type": str(row.get('issue_type', '')),
                "organic_recommendation": str(row.get('organic_recommendation', '')),
                "active_ingredients": str(row.get('active_ingredient_or_biological_agent', '')),
                "application_guidance": str(row.get('application_guidance', '')),
                "dosage": str(row.get('dosage', '')),
                "frequency": str(row.get('frequency', '')),
                "integrated_prevention": str(row.get('integrated_prevention', '')),
                "safety_precautions": str(row.get('safety_precautions', '')),
                "reference_to_verify": str(row.get('reference_to_verify', ''))
            })
        return results

# Singleton instance
recommender = PesticideRecommender()
