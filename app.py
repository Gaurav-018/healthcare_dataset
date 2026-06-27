import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Strict feature schema ordering matching the binary properties[cite: 1, 2]
FEATURE_NAMES = [
    "Age", "Gender", "Blood Type", "Medical Condition", 
    "Hospital", "Insurance Provider", "Billing Amount", 
    "Admission Type", "Medication"
]

# Smart file scanner to handle any of your pickle file names dynamically[cite: 1, 2]
POSSIBLE_MODEL_NAMES = ['model.pkl', 'XGBoost.pkl', 'xgboost.pkl']
model = None

for filename in POSSIBLE_MODEL_NAMES:
    path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
                print(f" -> Successfully loaded model file: {filename}")
                break
        except Exception as e:
            print(f" -> Error reading {filename}: {e}")

if model is None:
    print(" -> CRITICAL ALERT: No valid model pickle file found in repository root directory!")

# Attractive modern HTML UI definition
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XGBoost Prediction Engine</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            --panel-bg: rgba(30, 41, 59, 0.7);
            --border-color: rgba(255, 255, 255, 0.1);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #6366f1;
            --accent-hover: #4f46e5;
            --success: #10b981;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        body { background: var(--bg-gradient); color: var(--text-main); min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 2rem; }
        
        .container { background: var(--panel-bg); backdrop-filter: blur(16px); border: 1px solid var(--border-color); border-radius: 24px; width: 100%; max-width: 800px; padding: 2.5rem; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5); }
        header { text-align: center; margin-bottom: 2.5rem; }
        header h1 { font-size: 2.2rem; font-weight: 700; background: linear-gradient(to right, #a5b4fc, #6366f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }
        header p { color: var(--text-muted); font-size: 1rem; }

        .feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
        .input-group { display: flex; flex-direction: column; gap: 0.5rem; }
        .input-group label { font-size: 0.875rem; font-weight: 500; color: #cbd5e1; }
        
        .input-group input, .input-group select { background: rgba(15, 23, 42, 0.6); border: 1px solid var(--border-color); border-radius: 12px; padding: 0.75rem 1rem; color: var(--text-main); font-size: 1rem; transition: all 0.3s ease; outline: none; }
        .input-group input:focus, .input-group select:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2); }
        
        .btn-submit { background: var(--accent); color: white; border: none; border-radius: 12px; padding: 1rem; font-size: 1.1rem; font-weight: 600; width: 100%; cursor: pointer; transition: all 0.2s ease; display: flex; align-items: center; justify-content: center; gap: 0.5rem; }
        .btn-submit:hover { background: var(--accent-hover); transform: translateY(-1px); }
        .btn-submit:active { transform: translateY(0); }

        .result-panel { margin-top: 2rem; padding: 1.5rem; background: rgba(15, 23, 42, 0.4); border-radius: 16px; border: 1px solid var(--border-color); display: none; opacity: 0; transition: opacity 0.4s ease; }
        .result-panel.show { display: block; opacity: 1; }
        .result-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem; font-size: 1.25rem; font-weight: 600; color: var(--success); }
        .result-body { color: #e2e8f0; font-size: 1rem; line-height: 1.6; }
        .probability-tag { background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); padding: 0.25rem 0.5rem; border-radius: 6px; font-family: monospace; color: #34d399; }
    </style>
</head>
<body>

<div class="container">
    <header>
        <h1><i class="fa-solid fa-microchip"></i> Inference Engine Portal</h1>
        <p>Input real-time categorical parameters for predictive classification mapping</p>
    </header>

    <form id="predictionForm">
        <div class="feature-grid">
            <div class="input-group">
                <label for="Age">Age (Years)</label>
                <input type="number" id="Age" name="Age" placeholder="e.g. 45" required min="0" max="120">
            </div>
            <div class="input-group">
                <label for="Gender">Gender</label>
                <select id="Gender" name="Gender" required>
                    <option value="0">Male (0)</option>
                    <option value="1">Female (1)</option>
                </select>
            </div>
            <div class="input-group">
                <label for="Blood Type">Blood Type</label>
                <input type="number" id="Blood Type" name="Blood Type" placeholder="Numeric Index Value" required>
            </div>
            <div class="input-group">
                <label for="Medical Condition">Medical Condition</label>
                <input type="number" id="Medical Condition" name="Medical Condition" placeholder="Numeric Index Value" required>
            </div>
            <div class="input-group">
                <label for="Hospital">Hospital Identification</label>
                <input type="number" id="Hospital" name="Hospital" placeholder="Numeric Index Value" required>
            </div>
            <div class="input-group">
                <label for="Insurance Provider">Insurance Provider</label>
                <input type="number" id="Insurance Provider" name="Insurance Provider" placeholder="Numeric Index Value" required>
            </div>
            <div class="input-group">
                <label for="Billing Amount">Billing Amount ($)</label>
                <input type="number" step="0.01" id="Billing Amount" name="Billing Amount" placeholder="e.g. 14500.50" required>
            </div>
            <div class="input-group">
                <label for="Admission Type">Admission Type</label>
                <input type="number" id="Admission Type" name="Admission Type" placeholder="Numeric Index Value" required>
            </div>
            <div class="input-group">
                <label for="Medication">Medication Class</label>
                <input type="number" id="Medication" name="Medication" placeholder="Numeric Index Value" required>
            </div>
        </div>

        <button type="submit" class="btn-submit">
            <i class="fa-solid fa-wand-magic-sparkles"></i> Process Machine Prediction
        </button>
    </form>

    <div class="result-panel" id="resultPanel">
        <div class="result-header" id="resultHeader">
            <i class="fa-solid fa-circle-check"></i> Analysis Complete
        </div>
        <div class="result-body" id="resultBody"></div>
    </div>
</div>

<script>
    document.getElementById('predictionForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const payload = {
            "Age": parseInt(document.getElementById('Age').value),
            "Gender": parseInt(document.getElementById('Gender').value),
            "Blood Type": parseInt(document.getElementById('Blood Type').value),
            "Medical Condition": parseInt(document.getElementById('Medical Condition').value),
            "Hospital": parseInt(document.getElementById('Hospital').value),
            "Insurance Provider": parseInt(document.getElementById('Insurance Provider').value),
            "Billing Amount": parseFloat(document.getElementById('Billing Amount').value),
            "Admission Type": parseInt(document.getElementById('Admission Type').value),
            "Medication": parseInt(document.getElementById('Medication').value)
        };

        const panel = document.getElementById('resultPanel');
        const header = document.getElementById('resultHeader');
        const body = document.getElementById('resultBody');

        // Reveal the panel with an ongoing spinning visual state status
        panel.style.display = 'block';
        panel.classList.add('show');
        header.style.color = '#6366f1';
        header.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing Matrix Inference...';
        body.innerHTML = '';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            // Standard safe native object decoding parsing structure
            const result = await response.json(); 
            
            if (response.ok) {
                header.style.color = '#10b981';
                header.innerHTML = '<i class="fa-solid fa-circle-check"></i> Prediction Complete';
                body.innerHTML = `
                    <p style="margin-bottom: 0.75rem;"><strong>Predicted Class Designation:</strong> <span style="font-size: 1.25rem; color: #10b981; font-weight: bold;">Class ${result.prediction}</span></p>
                    <p style="margin-bottom: 0.25rem;"><strong>Confidence Probabilities Mapping:</strong></p>
                    <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.5rem;">
                        ${result.probabilities.map((p, idx) => `<span class="probability-tag">Idx ${idx}: ${(p * 100).toFixed(2)}%</span>`).join('')}
                    </div>
                `;
            } else {
                header.style.color = '#ef4444';
                header.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Model Error Context';
                body.innerHTML = `<p style="color: #ef4444;"><strong>Server Status Code ${response.status}:</strong> ${result.error || 'The model was unable to parse input variables.'}</p>`;
            }
        } catch (error) {
            console.error("Fetch API transaction crash:", error);
            header.style.color = '#ef4444';
            header.innerHTML = '<i class="fa-solid fa-wifi"></i> Connectivity Blocked';
            body.innerHTML = `<p style="color: #ef4444;"><strong>Error Context:</strong> Cannot communicate with the server endpoint. If this is a initial spin-up on a Free Instance, please wait 1-2 minutes for the system container to launch fully and refresh.</p>`;
        }
    });
</script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def portal_interface():
    return render_template_string(HTML_LAYOUT)

@app.route('/predict', methods=['POST'])
def run_predict_inference():
    if not model:
        return jsonify({"error": "The background engine workspace could not find or access your .pkl model file."}), 500
        
    try:
        data = request.get_json(force=True)
        
        # Verify complete payload matching against schema requirements
        features = []
        for feature in FEATURE_NAMES:
            if feature not in data:
                return jsonify({"error": f"Missing payload attribute parameter dimension: '{feature}'"}), 400
            features.append(data[feature])
            
        # Parse vector arrays smoothly into native arrays
        input_array = np.array([features])
        
        # Execute matrix inference[cite: 1, 2]
        prediction = model.predict(input_array)
        probabilities = model.predict_proba(input_array)
        
        return jsonify({
            "prediction": int(prediction[0]),
            "probabilities": probabilities[0].tolist()
        })
        
    except Exception as e:
        return jsonify({"error": f"Runtime computational pipeline failure: {str(e)}"}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
