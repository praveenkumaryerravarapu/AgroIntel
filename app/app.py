from flask import Flask, render_template, request, redirect
from markupsafe import Markup
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import io
import os
from PIL import Image
import joblib  # for loading/saving ML models
import torch
from torchvision import transforms
from utils.model import ResNet9


# ------------------------- DIRECTORY SETTINGS -------------------------------------------------------
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_APP_DIR)

print(f"--- Starting AgriSens Backend ---")
print(f"App Directory: {_APP_DIR}")


# ------------------------- CONFIGURATION & CONSTANTS ------------------------------------------------
disease_classes = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]


# ------------------------- LOADING THE TRAINED MODELS -----------------------------------------------

# Disease model loading
disease_model_path = os.path.join(_APP_DIR, 'models', 'plant_disease_model.pth')
disease_model = None

print(f"Loading Disease Model from: {disease_model_path}")
if os.path.exists(disease_model_path):
    try:
        # Load the architecture
        disease_model = ResNet9(3, len(disease_classes))
        # Load the weights (CPU mode for stability)
        disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
        disease_model.eval()
        print("✅ Disease Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading disease model: {e}")
        disease_model = None
else:
    print(f"⚠️ Warning: Disease model file not found at {disease_model_path}")

# crop recommendation model (trained with RandomForestClassifier)
crop_model = None
crop_model_path = os.path.join(_APP_DIR, 'models', 'crop_model.pkl')
print(f"Loading Crop Model from: {crop_model_path}")

if os.path.exists(crop_model_path):
    try:
        crop_model = joblib.load(crop_model_path)
        print("✅ Crop Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading crop model: {e}")
        crop_model = None
else:
    print(f"⚠️ Warning: Crop model file not found at {crop_model_path}")
    crop_model = None

# human‑readable crop names and categories for the recommendation output
CROP_INFO = {
    # Cereals
    'wheat': ('Wheat', 'Cereal'),
    'rice': ('Rice', 'Cereal'),
    'maize': ('Maize', 'Cereal'),
    'barley': ('Barley', 'Cereal'),
    'millet': ('Millet', 'Cereal'),
    # Pulses
    'chickpea': ('Chickpea', 'Pulse'),
    'lentil': ('Lentil', 'Pulse'),
    'pigeon_pea': ('Pigeon Pea', 'Pulse'),
    'green_gram': ('Green Gram (Moong)', 'Pulse'),
    'black_gram': ('Black Gram (Urad)', 'Pulse'),
    # Vegetables
    'tomato': ('Tomato', 'Vegetable'),
    'potato': ('Potato', 'Vegetable'),
    'onion': ('Onion', 'Vegetable'),
    'cabbage': ('Cabbage', 'Vegetable'),
    'cauliflower': ('Cauliflower', 'Vegetable'),
    'spinach': ('Spinach', 'Vegetable'),
    'carrot': ('Carrot', 'Vegetable'),
    'brinjal': ('Brinjal (Eggplant)', 'Vegetable'),
    'capsicum': ('Capsicum', 'Vegetable'),
    # Fruits
    'mango': ('Mango', 'Fruit'),
    'banana': ('Banana', 'Fruit'),
    'apple': ('Apple', 'Fruit'),
    'orange': ('Orange', 'Fruit'),
    'papaya': ('Papaya', 'Fruit'),
    'pomegranate': ('Pomegranate', 'Fruit'),
    'grapes': ('Grapes', 'Fruit'),
    'watermelon': ('Watermelon', 'Fruit'),
    'muskmelon': ('Muskmelon', 'Fruit'),
    # Oil crops
    'mustard': ('Mustard', 'Oil Crop'),
    'groundnut': ('Groundnut', 'Oil Crop'),
    'soybean': ('Soybean', 'Oil Crop'),
    'sunflower': ('Sunflower', 'Oil Crop'),
}

# simple growing tips per crop (can be expanded with more detailed agronomy data)
CROP_TIPS = {
    'wheat': "Prefers cool, dry climate with well‑drained loamy soil. Avoid waterlogging during grain filling.",
    'rice': "Needs warm temperatures and plenty of water; thrives in clayey or loamy soils with good puddling.",
    'maize': "Grows well in warm conditions with moderate rainfall and fertile, well‑drained soil.",
    'barley': "Tolerates cool, dry climates; grows best in well‑drained light to medium soils.",
    'millet': "Highly drought tolerant; suitable for light, sandy to loamy soils in low‑rainfall areas.",
    'chickpea': "Requires cool, dry weather and well‑drained loamy soil with neutral to slightly alkaline pH.",
    'lentil': "Prefers cool climate and moderately fertile, well‑drained loam; avoid heavy waterlogging.",
    'pigeon_pea': "Suited to warm, semi‑arid climates and deep, well‑drained loamy soils.",
    'green_gram': "Grows well in warm, humid climates on light loamy soils with good drainage.",
    'black_gram': "Performs best in warm, humid conditions and fertile, well‑drained soils.",
    'tomato': "Needs warm days, cool nights, and fertile, well‑drained soil rich in organic matter.",
    'potato': "Prefers cool climate and loose, well‑drained sandy loam; avoid high temperatures at tuberization.",
    'onion': "Requires cool weather for vegetative growth and warm, dry conditions for bulb development.",
    'cabbage': "Thrives in cool climate with fertile, well‑drained soil rich in organic matter.",
    'cauliflower': "Needs cool, moist climate and fertile, well‑drained loamy soil with good moisture.",
    'spinach': "Grows best in cool weather on fertile, well‑drained loam rich in nitrogen.",
    'carrot': "Prefers deep, loose, sandy loam soils free of stones for straight root development.",
    'brinjal': "Requires warm climate and fertile, well‑drained soil; sensitive to frost.",
    'capsicum': "Does well in cool to moderately warm climate and rich, well‑drained soils.",
    'mango': "Grows in tropical to subtropical climates on deep, well‑drained loamy soils.",
    'banana': "Needs warm, humid climate and rich, well‑drained soils with assured irrigation.",
    'apple': "Thrives in cool to temperate climates with well‑drained loamy soils and chilling hours.",
    'orange': "Prefers subtropical climate and well‑drained sandy loam; sensitive to waterlogging.",
    'papaya': "Requires warm climate and well‑drained, fertile soil; sensitive to frost and water stagnation.",
    'pomegranate': "Tolerates semi‑arid climate and light, well‑drained soils with low to moderate rainfall.",
    'grapes': "Needs warm, dry climate and deep, well‑drained soils; avoid heavy clay with poor drainage.",
    'watermelon': "Requires hot, dry climate and sandy loam soils with good drainage.",
    'muskmelon': "Grows well in warm, dry climates on light, well‑drained sandy loam.",
    'mustard': "Prefers cool season and fertile, well‑drained loam; sensitive to high temperatures at flowering.",
    'groundnut': "Needs warm climate and loose, sandy loam soils for proper pod development.",
    'soybean': "Grows well in warm, humid climates with fertile, well‑drained loams rich in organic matter.",
    'sunflower': "Tolerates semi‑arid climate and performs well on a wide range of well‑drained soils.",
}


def predict_image(img, model=disease_model):
    """
    Predict plant disease from image bytes using the loaded PyTorch model.
    """
    if model is None:
        # Fallback to hash-based demo mode if model isn't loaded
        idx = hash(img) % len(disease_classes)
        return disease_classes[idx]
        
    try:
        # Preprocessing exactly as per training script
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        image = Image.open(io.BytesIO(img)).convert('RGB')
        img_t = transform(image)
        img_u = torch.unsqueeze(img_t, 0)
        
        # Inference
        with torch.no_grad():
            yb = model(img_u)
            _, preds = torch.max(yb, dim=1)
            prediction = disease_classes[preds[0].item()]
            
        return prediction
    except Exception as e:
        print(f"Prediction Error: {e}")
        return None


app = Flask(__name__)


@app.route('/')
def home():
    title = 'Harvestify - Home'
    return render_template('index.html', title=title)


@app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Harvestify - Fertilizer Suggestion'
    
    # Load unique crops from the CSV to make the dropdown dynamic
    fert_csv_path = os.path.join(_APP_DIR, 'Data', 'fertilizer.csv')
    df = pd.read_csv(fert_csv_path)
    crops = sorted(df['Crop'].unique().tolist())
    
    # optional crop passed from crop recommendation page (?crop=rice)
    preselected_crop = request.args.get('crop')
    return render_template('fertilizer.html', title=title, crops=crops, preselected_crop=preselected_crop)


@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'

    crop_name = str(request.form['cropname']).strip()
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])

    fert_csv_path = os.path.join(_APP_DIR, 'Data', 'fertilizer.csv')
    df = pd.read_csv(fert_csv_path)

    # ensure crop exists in dataset
    if crop_name not in df['Crop'].values:
        return render_template(
            'fertilizer.html',
            title=title,
            error=f"No fertilizer data available for crop '{crop_name}'. Please choose another crop.",
            preselected_crop=None,
        )

    row = df[df['Crop'] == crop_name].iloc[0]
    nr = int(row['N'])
    pr = int(row['P'])
    kr = int(row['K'])

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if disease_model is None:
            return render_template(
                'disease.html',
                title=title,
                error=(
                    "Disease detection isn't available right now on this setup "
                    "(missing or incompatible PyTorch/model file). "
                    "The rest of the app will still work."
                ),
            )
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()
            prediction = predict_image(img)
            if prediction is None:
                return render_template('disease.html', title=title)
            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except Exception:
            pass
    # On GET, just show the page.
    return render_template('disease.html', title=title)


# ---- Crop recommendation endpoints ------------------------------------------------
@app.route('/crop')
def crop_recommendation():
    """Render form for entering soil/climate parameters."""
    print("crop_recommendation called")
    title = 'Harvestify - Crop Recommendation'
    return render_template('crop.html', title=title)


@app.route('/predict-crop', methods=['POST'])
def predict_crop():
    """Take form values, run the trained classifier, and show results."""
    title = 'Harvestify - Crop Recommendation'

    # validate inputs
    try:
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorous'])
        K = float(request.form['pottasium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph_val = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
    except Exception:
        return render_template('crop.html', title=title, error="Please enter valid numeric values for all fields.")

    if crop_model is None:
        return render_template('crop.html', title=title, error="Crop recommendation model not loaded. Please train the model first.")

    input_vector = [[N, P, K, temperature, humidity, ph_val, rainfall]]
    try:
        # main prediction
        predicted_label = crop_model.predict(input_vector)[0]

        top_crops = None
        confidence = None
        if hasattr(crop_model, 'predict_proba'):
            proba = crop_model.predict_proba(input_vector)[0]
            classes = list(crop_model.classes_)
            # sort classes by probability, descending
            class_probs = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)
            top_crops = []
            for label, p in class_probs[:3]:
                display_name, category = CROP_INFO.get(label, (label, 'Other'))
                top_crops.append(
                    {
                        'label': label,
                        'name': display_name,
                        'category': category,
                        'probability': p,
                    }
                )
            # confidence of the best prediction
            confidence = class_probs[0][1]

        # map best label to human‑readable name and category
        display_name, category = CROP_INFO.get(predicted_label, (predicted_label, 'Other'))
        tip = CROP_TIPS.get(predicted_label)

        return render_template(
            'crop-result.html',
            crop=display_name,
            crop_label=predicted_label,
            confidence=confidence,
            category=category,
            top_crops=top_crops,
            tip=tip,
            title=title,
        )
    except Exception as e:
        return render_template('crop.html', title=title, error=str(e))


if __name__ == '__main__':
    # show available routes for debugging
    print("Registered routes:")
    for rule in app.url_map.iter_rules():
        print(rule)
    app.run(debug=True)

