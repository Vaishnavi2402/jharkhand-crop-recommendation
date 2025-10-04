from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json

app = Flask(__name__)
CORS(app)

# Load artifacts
model = joblib.load('/content/crop_model_rf.pkl')
scaler = joblib.load('/content/scaler.pkl')
le_irrig = joblib.load('/content/le_irrig.pkl')
le_season = joblib.load('/content/le_season.pkl')
with open('/content/feature_cols.json','r') as f:
    feature_cols = json.load(f)

@app.route('/')
def home():
    return "Crop Recommendation API - Colab Test"

def prepare_input(data_json):
    # expect JSON keys matching subset of features, fallback to reasonable defaults if missing
    # Order must match feature_cols used in training
    vals = []
    # read raw numeric inputs; if missing use average or safe default
    for c in feature_cols:
        if c in data_json:
            vals.append(data_json[c])
        else:
            # simple fallback: zeros/mean-like defaults (you can refine)
            if 'Soil' in c or 'Rainfall' in c or 'Temperature' in c or 'Humidity' in c or 'Landholding' in c:
                vals.append(0)
            else:
                vals.append(0)
    return np.array(vals, dtype=float).reshape(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # map irrigation & season if present as strings
    if 'Irrigation' in data:
        try:
            data['Irrigation_enc'] = int(le_irrig.transform([data['Irrigation']])[0])
        except:
            return jsonify({"error":"Irrigation value not recognised. Options: "+str(list(le_irrig.classes_))}), 400
    if 'Season' in data:
        try:
            data['Season_enc'] = int(le_season.transform([data['Season']])[0])
        except:
            return jsonify({"error":"Season value not recognised. Options: "+str(list(le_season.classes_))}), 400

    # feature engineering same as training
    if 'Soil_Fertility' not in data:
        data['Soil_Fertility'] = data.get('Nitrogen_N',0) + data.get('Phosphorus_P',0) + data.get('Potassium_K',0)

    # Prepare vector in same order as feature_cols
    x = []
    for c in feature_cols:
        x.append(float(data.get(c, 0)))
    x = np.array(x).reshape(1,-1)
    x_scaled = scaler.transform(x)

    # get top 3
    probs = model.predict_proba(x_scaled)[0]
    classes = model.classes_
    top_idx = np.argsort(probs)[::-1][:3]
    recommendations = [{"Crop": classes[i], "Probability": float(round(probs[i],3))} for i in top_idx]

    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
