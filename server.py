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
@app.route("/")
def home():
    return "SmartBite Backend Running"

def enhance_for_ocr(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        denoised,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,11,2
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
"macros": {{"calories": "kcal","protein":"g","carbs":"g","fats":"g","fiber":"g"}},
"risks": ["point1 in {lang}", "point2 in {lang}"],
"alternatives": ["alt1 in {lang}", "alt2 in {lang}"],
"tips": ["tip1 in {lang}", "tip2 in {lang}"]
}}
"""

    try:
        res = client.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role":"system","content":f"You are a clinical nutrition expert fluent in {lang}."},
                {"role":"user","content":prompt}
            ]
        )
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        print("Groq Error:",e)
        return {"foodName":food,"status":"Error","explanation":"Analysis failed"}

# ====================================
# 2. RECIPES ENGINE
# ====================================
@app.route("/recipes", methods=["POST"])
def recipes():
    try:
        data = request.json
        ingredients = data.get("ingredients","")
        profile = data.get("profile",{})
        lang = profile.get("language","en")
        conditions = ", ".join(profile.get("conditions", []))

        prompt = f"""
        You are SmartBite AI Chef. Respond in {lang}.

        STYLE:
        - Use Massy local slang (Mama/Bangaram).
        - Energetic and helpful tone.

        DISH SELECTION RULES:
        1. Nuvvu ichina ingredients batti EXACTLY 3 DIFFERENT DISH OPTIONS suggest cheyyali.
        2. For each dish, provide a detailed 8-10 step procedure.
        3. Choose dishes that are safe for the user's medical condition: {conditions}.

        Return ONLY a JSON object with a 'recipes' key containing a list of 3 dishes:
        {{
          "recipes": [
            {{
              "name": "Dish Name",
              "why": "Health benefit line",
              "ingredients": ["item1", "item2"],
              "procedure": "Detailed 8-10 steps"
            }}
          ],
          "intro": "Idigo mama/bangaram, nee daggara unna items tho ee 3 racha dishes cheyochu. Edi kavalo select chesko!"
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
        print("Recipe Error:",e)
        return jsonify({"recipes":[], "intro": "System busy mama!"})

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
        gender = profile.get("gender", "male")
        conditions = ", ".join(profile.get("conditions", []))

        system_msg = f"""You are SmartBite AI. Respond in {lang}. 
        User Gender: {gender}.
        Conditions: {conditions}.

        STYLE RULES:
        1. If Male: Use 'Mama', 'Bhaiya'. If Female: Use 'Bangaram', 'Chelli'.
        2. Use Massy local slang (e.g., 'Gattiga', 'Sakkaga', 'Racha').
        3. Keep it very short. Use Bullet points. No long paragraphs."""

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
        return jsonify({"text":"System busy mama!"})

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
        print("OCR Error:",e)
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
        print("Plate Detect Error:",e)
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

    return jsonify(groq_analyze(profile,text,lang))

# ====================================
# SERVER START
# ====================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port)
