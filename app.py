# app.py

import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import requests
from io import BytesIO

# ---------------- Configuration ----------------
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
TOP_K = 5
CATEGORY_THRESHOLDS = {
    'Shoes': 0.55,
    'Bags': 0.50,
    'Watches': 0.60,
    'Glasses': 0.58
}
DEFAULT_THRESHOLD = 0.55  # fallback if category is unknown
FEATURE_FOLDER = 'data'
CSV_PATH = 'Database_Images/products.csv'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- Flask App ----------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------------- Load Dataset and Features ----------------
df = pd.read_csv(CSV_PATH)

# Ensure required columns exist
required_columns = ['Name', 'Category', 'price', 'image_path']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' missing in CSV!")

# Load precomputed features
feature_array = np.load(os.path.join(FEATURE_FOLDER, 'normalized_features.npy'))
image_paths = np.load(os.path.join(FEATURE_FOLDER, 'image_paths.npy'))

# ---------------- Load ResNet50 Model ----------------
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
print("ResNet50 model loaded.")

# ---------------- Utility Functions ----------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(img_path, model):
    """Extract ResNet50 features from an image."""
    try:
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def find_similar_images(uploaded_image_path, model, features, image_paths, top_k=TOP_K):
    """Return top K similar images and their scores."""
    uploaded_features = extract_features(uploaded_image_path, model)
    if uploaded_features is None:
        return []
    similarities = cosine_similarity([uploaded_features], features)[0]
    top_indices = heapq.nlargest(top_k, range(len(similarities)), key=lambda i: similarities[i])
    return [(image_paths[i], similarities[i]) for i in top_indices]

def download_image_from_url(url, save_folder):
    """Download image from URL and save it locally."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        response = requests.get(url, headers=headers, stream=True, timeout=10)
        response.raise_for_status()  # Raise error if status is not 200
        img = Image.open(BytesIO(response.content)).convert("RGB")
        filename = secure_filename(url.split("/")[-1])
        if filename == '':
            filename = "query_image.jpg"
        save_path = os.path.join(save_folder, filename)
        img.save(save_path)
        return save_path
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

# ---------------- Routes ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files.get('file')
    image_url = request.form.get('image_url')

    # Save uploaded file or download URL
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
    elif image_url:
        save_path = download_image_from_url(image_url, app.config['UPLOAD_FOLDER'])
        if save_path is None:
            return "Failed to download image from URL.", 400
    else:
        return "No file or URL provided.", 400

    # Find top similar images
    top_similar = find_similar_images(save_path, base_model, feature_array, image_paths, TOP_K)

    # Normalize CSV image paths for comparison
    df['image_path'] = df['image_path'].str.replace("\\", "/").str.strip()

    results = []
    query_image_rel = os.path.relpath(save_path, 'static').replace("\\", "/")

    # Category-aware threshold filtering
    for img_path, score in top_similar:
        # Make img_path relative to static/ folder
        if os.path.isabs(img_path):
            rel_path = os.path.relpath(img_path, 'static').replace("\\", "/")
        else:
            rel_path = img_path.replace("\\", "/")

        # Match CSV using cleaned path
        row = df[df['image_path'] == rel_path]

        # Fallback: match by filename only if path fails
        if row.empty:
            filename_only = os.path.basename(rel_path)
            row = df[df['image_path'].str.endswith(filename_only)]

        if not row.empty:
            row = row.iloc[0]
            category = row['Category']
            threshold = CATEGORY_THRESHOLDS.get(category, DEFAULT_THRESHOLD)

            # Only append if similarity passes the category threshold
            if score >= threshold:
                results.append({
                    'name': row['Name'],
                    'Category': row['Category'],
                    'price': row['price'],
                    'image_path': row['image_path'],
                    'score': score
                })

    # Handle no match case
    no_match = False
    if not results:
        no_match = True

    # Debug prints
    print("Query image:", query_image_rel)
    for r in results:
        print("Result image:", r['image_path'])

    return render_template('index.html', query_image=query_image_rel, results=results, no_match=no_match)

# ---------------- Run App ----------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
