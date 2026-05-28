import os
import joblib
import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except Exception:
    ort = None


class ModelService:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.image_model = None
        self.image_session = None
        self.tabular_model = None
        self.price_scaler = None
        self.encoders = None
        self._load_models()

    def _load_models(self):
        # Load image model
        image_path = os.path.join(self.models_dir, 'image_model.onnx')
        if ort is not None and os.path.exists(image_path):
            self.image_session = ort.InferenceSession(image_path, providers=['CPUExecutionProvider'])

        # Load tabular artifacts
        scaler_path = os.path.join(self.models_dir, 'price_scaler.pkl')
        model_path = os.path.join(self.models_dir, 'model.pkl')
        encoders_path = os.path.join(self.models_dir, 'encoders.pkl')

        if os.path.exists(scaler_path):
            self.price_scaler = joblib.load(scaler_path)
        if os.path.exists(model_path):
            self.tabular_model = joblib.load(model_path)
        if os.path.exists(encoders_path):
            self.encoders = joblib.load(encoders_path)

    def predict_image(self, image_path):
        if self.image_session is None:
            raise RuntimeError('Image model not available in this deployment')
        try:
            img = Image.open(image_path).convert('RGB').resize((100, 100))
        except Exception:
            raise ValueError('Invalid image')
        img = np.asarray(img, dtype='float32') / 255.0
        img = np.expand_dims(img, axis=0)
        input_name = self.image_session.get_inputs()[0].name
        outputs = self.image_session.run(None, {input_name: img})
        scaled_pred = float(np.asarray(outputs[0]).reshape(-1)[0])
        if self.price_scaler is not None:
            inverse = self.price_scaler.inverse_transform([[scaled_pred]])
            actual_price = float(inverse[0][0])
        else:
            actual_price = scaled_pred
        return {'price': actual_price, 'raw': scaled_pred}

    def predict_tabular(self, features: list):
        if self.tabular_model is None:
            raise RuntimeError('Tabular model not loaded')
        pred = float(self.tabular_model.predict([features])[0])
        return {'price': pred}
