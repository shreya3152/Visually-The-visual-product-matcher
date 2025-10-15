# Visually - The Visual Product Matcher

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3-green)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

**Visually** is a web application that allows users to upload an image or provide an image URL to find visually similar products from a curated database. The app uses **deep learning (ResNet50)** for image feature extraction and **cosine similarity** for comparing images. It is ideal for e-commerce platforms, helping users quickly find products that match their visual preferences.

---

## Features

- Upload images from your device or provide an image URL.
- Automatically extracts features using ResNet50.
- Finds the top similar products from a database of images.
- Displays product name, category, price, and similarity score.
- Category-aware threshold filtering ensures only relevant matches are shown.
- Responsive UI with a sleek, modern design.

---

## Demo
The output of the file is diplayed separately in the output folder.



---

## Technologies Used

- **Python 3**
- **Flask**
- **TensorFlow / Keras (ResNet50)**
- **NumPy & Pandas**
- **Scikit-learn**
- **Bootstrap 5 & CSS3**
- **HTML5**

---

## Installation

Follow these steps to set up and run the **Visually** application locally:

1. **Clone the repository:**
   ```
   git clone https://github.com/shreya3152/Visually-The-visual-product-matcher.git
   cd Visually-The-visual-product-matcher
   ```

2. **Create and activate a virtual environment:**
   - Windows:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
   
   - macOS/Linux:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Run the Flask application:**
   ```
   python app.py
   ```

5. **Open your browser and visit:**
   ```
   http://127.0.0.1:5000
   ```

---
## Usage

- Use the Upload from Device tab to select an image from your computer.
- Use the Upload via URL tab to provide a direct link to an image online.
- Click Search to see the top similar products.
- Results show product name, category, price, and similarity score.

---
## Contribution

- Contributions are welcome! You can help by:
- Adding more product categories.
- Improving feature extraction models.
- Enhancing UI/UX design.

---
## License

This project is licensed under the MIT License. See the LICENSE
 file for details.
