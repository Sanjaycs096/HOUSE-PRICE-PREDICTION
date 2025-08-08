from flask import Flask, request, render_template
import os
import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
image_model = load_model('models/image_model.h5', compile=False)                    # CNN image model
price_scaler = joblib.load('models/price_scaler.pkl')                # Scaler for price normalization
tabular_model = joblib.load('models/model.pkl')                      # Tabular ML model
label_encoders = joblib.load('models/encoders.pkl')                  # Encoders for categorical values

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    form_data = {}
    uploaded_image_url = None

    if request.method == 'POST':
        form_data = request.form.to_dict()
        image_file = request.files.get('image')

        if image_file and image_file.filename != '':
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            # Normalize image path for displaying in HTML
            uploaded_image_url = image_path.replace('static/', '', 1).replace("\\", "/")

            try:
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError("Uploaded image could not be read. Check file format or if it's corrupted.")

                img = cv2.resize(img, (100, 100)).astype('float32') / 255.0
                img = np.expand_dims(img, axis=0)

                scaled_pred = image_model.predict(img)[0][0]
                actual_price = price_scaler.inverse_transform([[scaled_pred]])[0][0]
                prediction = f"🏠 Predicted Price from image: ₹{actual_price:,.0f}"

            except Exception as e:
                prediction = f"❌ Image Prediction Error: {str(e)}"
                uploaded_image_url = image_file.filename

        else:
            try:
                # Apply label encoding to categorical fields
                location = label_encoders['Location'].transform([form_data.get("Location")])[0]
                property_type = label_encoders['Property_Type'].transform([form_data.get("Property_Type")])[0]
                furnishing = label_encoders['Furnishing'].transform([form_data.get("Furnishing")])[0]

                features = [
                    location,
                    float(form_data.get("Area")),
                    int(form_data.get("BHK")),
                    int(form_data.get("Bathrooms")),
                    property_type,
                    int(form_data.get("Age")),
                    furnishing,
                    int(form_data.get("Floor_Number")),
                    int(form_data.get("Total_Floors")),
                    int(form_data.get("Parking")),
                    int(form_data.get("proximity"))
                ]

                pred = tabular_model.predict([features])[0]
                prediction = f"🏠 Predicted Price from form: ₹{pred:,.0f}"

            except Exception as e:
                prediction = f"❌ Form Prediction Error: {str(e)}"

    return render_template("predict.html", prediction=prediction, form_data=form_data, uploaded_image_url=uploaded_image_url)

if __name__ == "__main__":
    app.run(debug=True)
