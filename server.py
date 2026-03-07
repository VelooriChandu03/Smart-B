import os
import json
import re
import base64
import io
import requests
import numpy as np
import cv2

from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)

# ENV VARIABLES
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
ROBOFLOW_WORKFLOW_URL = os.environ.get("ROBOFLOW_WORKFLOW_URL")

MODEL = "llama-3.3-70b-versatile"
client = Groq(api_key=GROQ_API_KEY)

# ====================================
# ADVANCED OCR HELPERS (IMPROVED SCANNING)
# ====================================


def enhance_for_ocr(cv_img):
    """OCR Accuracy penche logic mama"""
    # 1. Grayscale conversion
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # 2. Resizing (2x) - Chinna text ni highlight chestundi
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # 3. Denoising - Blurry spots clean chestundi
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    # 4. Adaptive Thresholding - Lighting issues fix chestundi
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

def clean_text(t):
    return re.sub(r"\s+", " ", str(t)).strip()

def process_image(image_b64):
    if "," in image_b64:
        image_b64 = image_b64.split(",")[1]
    img_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ====================================
# 1. CORE ANALYSIS ENGINE (Language Specific)
# ====================================
def groq_analyze(profile, food, lang="en"):
    if not food or food.strip() == "":
        food = "Unknown Item"

    lang_instruction = f"IMPORTANT: Respond ONLY in {lang} language. If {lang} is Telugu, use Telugu script."

    prompt = f"""
    Role: Senior Clinical Dietitian.
    User Profile Conditions: {profile.get('conditions', 'General Health')}
    Food/Product: {food}
    Target Language: {lang}
    
    {lang_instruction}

    Return ONLY a JSON object:
    {{
      "foodName": "{food}",
      "status": "Safe/Caution/Unsafe",
      "explanation": "2-3 sentence clinical verdict in {lang}.",
      "healthScore": 1-10,
      "macros": {{"calories": "kcal", "protein": "g", "carbs": "g", "fats": "g", "fiber": "g"}},
      "risks": ["point1 in {lang}", "point2 in {lang}"],
      "alternatives": ["alt1 in {lang}", "alt2 in {lang}"],
      "tips": ["tip1 in {lang}", "tip2 in {lang}"]
    }}
    """
    try:
        res = client.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": f"You are a clinical nutrition expert fluent in {lang}."},
                      {"role": "user", "content": prompt}]
        )
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        print(f"Groq Analysis Error: {e}")
        return {"foodName": food, "status": "Error", "explanation": "Analysis failed."}

# ====================================
# 2. KITCHEN ENGINE (5 Recipes + Multi-lang)
# ====================================
@app.route("/recipes", methods=["POST"])
def recipes():
    try:
        data = request.json
        ingredients = data.get("ingredients", "")
        profile = data.get("profile", {})
        cuisine = data.get("cuisine", "Healthy")
        lang = profile.get("language", "en")
        
        prompt = f"""
        Role: Clinical Nutritionist and Professional Chef.
        Task: Create 5 healthy recipes in {cuisine} style.
        Ingredients: {ingredients}
        User Conditions: {profile.get('conditions')}
        Language: {lang}
        
        STRICT INSTRUCTIONS:
        1. All fields MUST be in {lang}.
        2. Instructions must be a list of clear, numbered steps.
        3. Mention heat levels and exact timing.

        Return ONLY a JSON object:
        {{
          "recipes": [
            {{
              "id": "unique_id",
              "title": "Name in {lang}",
              "ingredients": ["quantities in {lang}"],
              "instructions": ["Step 1 in {lang}", "Step 2 in {lang}"],
              "prepTime": "XX mins",
              "calories": "XXX",
              "whyItsGood": "Reason in {lang}"
            }}
          ]
        }}
        """
        res = client.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return res.choices[0].message.content
    except Exception as e:
        print(f"Recipe Error: {e}")
        return jsonify({"recipes": []})

# ====================================
# 3. CHAT ENGINE (Health Buddy)
# ====================================
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        msg = data.get("message")
        profile = data.get("profile", {})
        lang = profile.get("language", "en")
        
        system_msg = f"You are 'SmartBite AI', a health buddy. Respond ONLY in {lang}. Conditions: {profile.get('conditions')}."

        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user", "content": msg}]
        )
        return jsonify({"text": res.choices[0].message.content})
    except Exception as e:
        return jsonify({"text": "System busy, mama!"})

# ====================================
# 4. SCANNING (OCR & PLATE) - FULLY FIXED
# ====================================
@app.route("/ocr-analyze", methods=["POST"])
def ocr_analyze():
    try:
        data = request.json
        raw_img = process_image(data.get("imageB64"))
        
        # Applying Enhancement for better accuracy
        processed_img = enhance_for_ocr(raw_img)
        
        reader = get_ocr()
        # detail=0 for direct text strings
        results = reader.readtext(processed_img, detail=0)
        text = clean_text(" ".join(results))
        
        if not text:
            # Fallback to raw image if processed image fails
            results = reader.readtext(raw_img, detail=0)
            text = clean_text(" ".join(results))

        lang = data.get("profile", {}).get("language", "en")
        analysis = groq_analyze(data.get("profile", {}), text or "Food Label", lang)
        analysis["ocrText"] = text
        return jsonify(analysis)
    except Exception as e:
        print(f"OCR Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/plate-detect", methods=["POST"])
def plate_detect():
    try:
        data = request.json
        image_b64 = data.get("imageB64")
        if "," in image_b64: image_b64 = image_b64.split(",")[1]

        rf_res = requests.post(
            f"{ROBOFLOW_WORKFLOW_URL}?api_key={ROBOFLOW_API_KEY}",
            data=image_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10
        )
        rf_data = rf_res.json()
        
        preds = []
        if 'outputs' in rf_data and len(rf_data['outputs']) > 0:
            preds = rf_data['outputs'][0].get('predictions', [])
        elif 'predictions' in rf_data:
            preds = rf_data['predictions']
            
        detected_list = [p.get('class') for p in preds if p.get('confidence', 0) > 0.4]
        detected = ", ".join(list(set(detected_list))) if detected_list else "Healthy Plate"
        
        lang = data.get("profile", {}).get("language", "en")
        analysis = groq_analyze(data.get("profile", {}), detected, lang)
        analysis["detectedFoods"] = detected
        return jsonify(analysis)
    except Exception as e:
        print(f"Plate Detect Error: {e}")
        return jsonify({"error": "Detection failed"}), 500

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    profile = data.get("profile", {})
    text = data.get("text", "")
    lang = profile.get("language", "en")
    return jsonify(groq_analyze(profile, text, lang))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
