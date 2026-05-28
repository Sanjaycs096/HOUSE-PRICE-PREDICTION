import os
import joblib
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import InputLayer
    from tensorflow.keras.mixed_precision import Policy
    _TF_AVAILABLE = True
except Exception:
    load_model = None
    InputLayer = object
    Policy = object
    _TF_AVAILABLE = False


# Compatibility shims for legacy H5 models
class CompatibleInputLayer(InputLayer):
    @classmethod
    def from_config(cls, config):
        config = dict(config)
        if "batch_shape" in config and "batch_input_shape" not in config:
            config["batch_input_shape"] = config.pop("batch_shape")
        return super().from_config(config)


class CompatibleDTypePolicy(Policy):
    @classmethod
    def from_config(cls, config):
        return Policy(config.get("name", "float32"))


class ModelService:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.image_model = None
        self.tabular_model = None
        self.price_scaler = None
        self.encoders = None
        self._load_models()

    def _load_models(self):
        # Load image model
        image_path = os.path.join(self.models_dir, 'image_model.h5')
        if _TF_AVAILABLE and os.path.exists(image_path):
            self.image_model = load_model(
                image_path,
                compile=False,
                custom_objects={
                    "InputLayer": CompatibleInputLayer,
                    "DTypePolicy": CompatibleDTypePolicy,
                },
            )

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
        if self.image_model is None or cv2 is None:
            raise RuntimeError('Image model not available in this deployment')
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError('Invalid image')
        img = cv2.resize(img, (100, 100)).astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        scaled_pred = float(self.image_model.predict(img)[0][0])
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
