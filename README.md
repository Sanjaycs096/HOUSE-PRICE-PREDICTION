# 🏠 House Price Prediction (AI-Powered)

This project is an **AI-powered House Price Prediction web application** that estimates the price of a property based on various input features such as location, area, BHK, furnishing type, property type, and other attributes.  
It also allows image uploads to assist in prediction, making it user-friendly for both real estate agents and home buyers.

---

## 🚀 Features
- **Interactive Web UI** – Clean and modern design for smooth user experience.
- **Form-Based Prediction** – Enter property details to get instant price estimation.
- **Image Upload Support** – Upload an image of the property to assist prediction.
- **AI/ML Model Integration** – Backend model trained on real estate datasets.
- **Responsive Design** – Works on desktops, tablets, and mobile devices.

---

## 📸 Screenshots
### Prediction Form
![Prediction Form](assets/predict(1).png)(assets/predict(2).png)

### Home Page
![Home Page](assets/home.png)

---

## ⚙️ Technologies Used
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python (Flask / FastAPI)
- **Machine Learning**: Scikit-learn / TensorFlow / Pandas / NumPy
- **Deployment**: GitHub, Heroku / Render

---

## 📂 How to Run Locally
1. **Clone the repository**
   ```bash
   git clone https://github.com/Sanjaycs096/HOUSE-PRICE-PREDICTION.git
   cd HOUSE-PRICE-PREDICTION
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # For Linux/Mac
   .venv\Scripts\activate      # For Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   ```
   http://127.0.0.1:5000
   ```

---

## ⚠️ Limitation
> **Disadvantage**:  
> If a non-house image (such as an image of a tree, car, or unrelated object) is uploaded, the model **will still predict a house price** because it is not trained to detect whether the image is actually a house. This can lead to unrealistic outputs.

---

## 📜 License
This project is licensed under the MIT License – feel free to use and modify.

---

## 💡 Future Improvements
- Add **image classification** to verify if the uploaded image is actually a house before price prediction.
- Improve dataset quality with more diverse property images.
- Integrate live property market APIs for dynamic pricing.
