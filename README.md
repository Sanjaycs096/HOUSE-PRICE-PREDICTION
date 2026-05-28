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

## ⚙️ Technologies Used
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python (Flask / FastAPI)
- **Machine Learning**: Scikit-learn / TensorFlow / Pandas / NumPy
- **Deployment**: GitHub, Heroku / Render
- **Serverless Deployment**: Vercel (Python serverless functions)

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

   Vercel uses `pyproject.toml` and `api/index.py` directly, so the legacy `builds` and `functions` blocks are not needed.

   If you want full local image-model support on Windows, also install:
   ```bash
   pip install -r requirements-local.txt
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


## 💡 Future Improvements
- Add **image classification** to verify if the uploaded image is actually a house before price prediction.
- Improve dataset quality with more diverse property images.
- Integrate live property market APIs for dynamic pricing.

---

## 📦 Deploying to Vercel (Serverless)

This repository includes a small integration to run the Flask app on Vercel using a WSGI adapter.

Notes before deploying:
- Vercel imposes size and execution time limits. Large model files in `models/` (TensorFlow/Keras HDF5 files) may exceed Vercel limits and are not recommended to be deployed directly on Vercel. If your models are large, host them on a model-serving service (e.g., AWS SageMaker, Azure ML, or a small VM) and call that API from this app.
- The repository includes a `.python-version` file pinned to Python `3.12`, which matches Vercel's supported Python runtime.
- Vercel now reads `pyproject.toml` for the deploy-safe dependency list and `api/index.py` for the entrypoint; `vercel.json` only keeps the route mapping.
- TensorFlow and Keras are kept in `requirements-local.txt` for local development.
- Ensure `SECRET_KEY` and other sensitive environment variables are set in Vercel dashboard for the project.

Quick deployment steps:

1. Install the Vercel CLI and login:
   ```bash
   npm i -g vercel
   vercel login
   ```

2. Ensure `requirements.txt` includes `vercel-wsgi` (already added).

3. Create a new Vercel project and connect your GitHub repository, or deploy directly from CLI:
   ```bash
   vercel --prod
   ```

4. In the Vercel project settings, set environment variables (e.g. `SECRET_KEY`) and increase function memory/timeouts if available.

Troubleshooting:
- If Chart.js or other CDN assets are blocked by CSP, check `app.py` CSP settings; adjust allowed hosts in `vercel.json` or `app.py` if needed.
- If model files are too big: move model loading to an external API and update `services/model_service.py` to call the hosted model endpoint.

If you want, I can prepare a smaller demonstration model and a streamlined deploy branch that is Vercel-compatible (removes heavy model files and serves a lightweight stubbed model for demo). Would you like that?
