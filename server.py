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

# Language Mapping Helper
def get_language_full_name(lang_code):
    mapping = {
        "en": "English",
        "te": "Telugu (తెలుగు)",
        "hi": "Hindi (हिन्दी)",
        "ta": "Tamil (தமிழ்)",
        "kn": "Kannada (ಕನ್ನಡ)",
        "gu": "Gujarati (ગુજરાતી)",
        "bn": "Bengali (বাংলা)",
        "mr": "Marathi (మరాఠీ)"
    }
    return mapping.get(lang_code, "English")

# ====================================
# ADVANCED OCR HELPERS
# ====================================
@app.route("/")
def home():
    return "SmartBite Backend Running"

def enhance_for_ocr(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
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
# 1. CORE ANALYSIS ENGINE
# ====================================
def groq_analyze(profile, food, lang="en"):
    if not food or food.strip() == "":
        food = "Unknown Item"

    target_lang = get_language_full_name(lang)

    prompt = f"""
Role: Senior Clinical Dietitian.
User Profile Conditions: {profile.get('conditions', 'General Health')}
Food/Product: {food}
Target Language: {target_lang}

STRICT INSTRUCTION: Respond ONLY in {target_lang}. All descriptions and names must be in {target_lang} script.

Return ONLY a JSON object:
{{
"foodName": "Food name in {target_lang}",
"status": "Safe/Caution/Unsafe (Translated)",
"explanation": "2-3 sentence clinical verdict in {target_lang}.",
"healthScore": 1-10,
"macros": {{"calories": "kcal","protein":"g","carbs":"g","fats":"g","fiber":"g"}},
"risks": ["point1 in {target_lang}", "point2 in {target_lang}"],
"alternatives": ["alt1 in {target_lang}", "alt2 in {target_lang}"],
"tips": ["tip1 in {target_lang}", "tip2 in {target_lang}"]
}}
"""

    try:
        res = client.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role":"system","content":f"You are a clinical nutrition expert providing output in {target_lang}."},
                {"role":"user","content":prompt}
            ]
        )
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        print("Groq Error:", e)
        return {"foodName": food, "status": "Error", "explanation": "Analysis failed"}

# ====================================
# 2. RECIPES ENGINE
# ====================================
# ====================================
# 2. RECIPES ENGINE (UPDATED FOR 10 STEPS)
# ====================================
@app.route("/recipes", methods=["POST"])
def recipes():
    try:
        data = request.json
        ingredients = data.get("ingredients","")
        profile = data.get("profile",{})
        lang = profile.get("language","en")
        target_lang = get_language_full_name(lang)
        conditions = ", ".join(profile.get("conditions", []))

        prompt = f"""
Role: Professional Nutritionist & Chef.
Task: Suggest EXACTLY 5 professional recipes based on ingredients: {ingredients}.
Medical Safety: Must be safe for {conditions}.
STRICT LANGUAGE RULE: The entire response must be in {target_lang}.

STRICT PROCEDURE RULE: 
The "procedure" MUST contain EXACTLY 10 detailed steps, from preparation to final serving. 
Format the steps as: "1. Step description\\n2. Step description... up to 10."

Format: Return ONLY a JSON object.
{{
  "recipes": [
    {{
      "id": "unique_id_1",
      "name": "Recipe Name in {target_lang}",
      "why": "Clinical benefit in {target_lang}",
      "ingredients": ["item 1 in {target_lang}", "item 2 in {target_lang}"],
      "procedure": "1. [Step 1 description in {target_lang}]\\n2. [Step 2 description in {target_lang}]\\n...\\n10. [Final step in {target_lang}]",
      "calories": "350",
      "prepTime": "25 mins"
    }}
  ],
  "intro": "Personalized intro in {target_lang}."
}}
"""

        res = client.chat.completions.create(
            model=MODEL,
            response_format={"type":"json_object"},
            messages=[{"role":"user","content":prompt}],
            temperature=0.3
        )
        return jsonify(json.loads(res.choices[0].message.content))
    except Exception as e:
        print("Recipe Error:", e)
        return jsonify({"recipes":[], "intro": "Error generating recipes."})
# ====================================
# 3. CHAT ENGINE
# ====================================
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        msg = data.get("message")
        profile = data.get("profile",{})
        lang = profile.get("language","en")
        target_lang = get_language_full_name(lang)
        conditions = ", ".join(profile.get("conditions", []))

        system_msg = f"""You are SmartBite AI, a professional health assistant. 
        You MUST respond ONLY in {target_lang} script.
        Conditions: {conditions}.

        STYLE RULES:
        1. Be supportive, professional, and direct. 
        2. DO NOT use 'Sir', 'Madam', or any informal slang. 
        3. Maintain a neutral, helpful tone.
        4. Use {target_lang} for the entire conversation.
        5. Provide accurate, clinical-based nutrition advice in bullet points."""

        res = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role":"system","content":system_msg},
                {"role":"user","content":msg}
            ]
        )

        return jsonify({"text":res.choices[0].message.content})

    except Exception as e:
        print("Chat Error:", e)
        return jsonify({"text":"Service is currently unavailable."})

# ====================================
# 4. OCR ANALYZE
# ====================================
@app.route("/ocr-analyze", methods=["POST"])
def ocr_analyze():
    try:
        data = request.json
        text = data.get("text","Food Label")
        lang = data.get("profile",{}).get("language","en")
        analysis = groq_analyze(data.get("profile",{}), text, lang)
        analysis["ocrText"] = text
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error":str(e)}),500

# ====================================
# 5. PLATE DETECTION
# ====================================
@app.route("/plate-detect", methods=["POST"])
def plate_detect():
    try:
        data = request.json
        image_b64 = data.get("imageB64")
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]

        rf_res = requests.post(
            f"{ROBOFLOW_WORKFLOW_URL}?api_key={ROBOFLOW_API_KEY}",
            data=image_b64,
            headers={"Content-Type":"application/x-www-form-urlencoded"},
            timeout=10
        )
        rf_data = rf_res.json()

        preds = []
        if 'outputs' in rf_data and len(rf_data['outputs'])>0:
            preds = rf_data['outputs'][0].get('predictions',[])
        elif 'predictions' in rf_data:
            preds = rf_data['predictions']

        detected_list = [p.get('class') for p in preds if p.get('confidence',0)>0.4]
        detected = ", ".join(list(set(detected_list))) if detected_list else "Healthy Plate"

        lang = data.get("profile",{}).get("language","en")
        analysis = groq_analyze(data.get("profile",{}), detected, lang)
        analysis["detectedFoods"] = detected
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error":"Detection failed"}),500

# ====================================
# 6. TEXT ANALYSIS
# ====================================
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    profile = data.get("profile",{})
    text = data.get("text","")
    lang = profile.get("language","en")
    return jsonify(groq_analyze(profile, text, lang))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
