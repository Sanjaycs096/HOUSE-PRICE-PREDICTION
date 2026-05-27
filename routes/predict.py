from flask import Blueprint, request, jsonify, current_app, session, abort
from werkzeug.utils import secure_filename
import os
import uuid
from services.model_service import ModelService

predict_bp = Blueprint('predict_api', __name__)

# initialize a global model service on first request
_model_service = None


def get_model_service():
    global _model_service
    if _model_service is None:
        models_dir = current_app.config.get('MODELS_DIR', 'models')
        _model_service = ModelService(models_dir=models_dir)
    return _model_service


def _is_allowed_image(filename: str, mimetype: str) -> bool:
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    allowed_ext = current_app.config.get('ALLOWED_IMAGE_EXTENSIONS', set())
    allowed_prefix = current_app.config.get('ALLOWED_IMAGE_MIME_PREFIX', 'image/')
    return ext in allowed_ext and str(mimetype).startswith(allowed_prefix)


def _verify_csrf():
    token = session.get('csrf_token')
    header = request.headers.get('X-CSRF-Token')
    if not token or not header or token != header:
        current_app.logger.warning('CSRF token missing or invalid')
        abort(400, description='CSRF token missing or invalid')


@predict_bp.route('/tabular', methods=['POST'])
def tabular_predict():
    _verify_csrf()
    data = request.form.to_dict()
    try:
        encoders = get_model_service().encoders
        if encoders is None:
            raise RuntimeError('Encoders not loaded')
        location = data.get('Location') or ''
        property_type = data.get('Property_Type') or ''
        furnishing = data.get('Furnishing') or ''
        # Note: simplistic feature extraction — keep consistent with training
        # Validate numeric inputs
        try:
            area = float(data.get('Area', 0))
            bhk = int(float(data.get('BHK', 0)))
            bathrooms = int(float(data.get('Bathrooms', 0)))
            age = int(float(data.get('Age', 0)))
            floor_number = int(float(data.get('Floor_Number', 0)))
            total_floors = int(float(data.get('Total_Floors', 0)))
            parking = int(float(data.get('Parking', 0)))
            proximity = float(data.get('proximity', 0))
        except ValueError:
            return jsonify({'ok': False, 'error': 'Invalid numeric input'}), 400

        # Validate categorical inputs against encoder classes if available
        def valid_category(enc, val):
            try:
                classes = getattr(enc, 'classes_', None)
                if classes is None:
                    return True
                return val in classes
            except Exception:
                return False

        if not valid_category(encoders.get('Location'), location):
            return jsonify({'ok': False, 'error': 'Unsupported Location value'}), 400
        if not valid_category(encoders.get('Property_Type'), property_type):
            return jsonify({'ok': False, 'error': 'Unsupported Property_Type value'}), 400
        if not valid_category(encoders.get('Furnishing'), furnishing):
            return jsonify({'ok': False, 'error': 'Unsupported Furnishing value'}), 400

        features = [
            encoders['Location'].transform([location])[0],
            area,
            bhk,
            bathrooms,
            encoders['Property_Type'].transform([property_type])[0],
            age,
            encoders['Furnishing'].transform([furnishing])[0],
            floor_number,
            total_floors,
            parking,
            proximity,
        ]
        res = get_model_service().predict_tabular(features)
        return jsonify({'ok': True, 'result': res})
    except Exception:
        current_app.logger.exception('Tabular prediction failed')
        return jsonify({'ok': False, 'error': 'Invalid or unsupported input'}), 400


@predict_bp.route('/image', methods=['POST'])
def image_predict():
    _verify_csrf()
    if 'image' not in request.files:
        return jsonify({'ok': False, 'error': 'No image file provided'}), 400

    f = request.files['image']
    if not f.filename:
        return jsonify({'ok': False, 'error': 'No image file provided'}), 400

    safe_name = secure_filename(f.filename)
    if not _is_allowed_image(safe_name, f.mimetype):
        return jsonify({'ok': False, 'error': 'Unsupported image type'}), 400

    ext = safe_name.rsplit('.', 1)[-1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'static/uploads')
    os.makedirs(upload_folder, exist_ok=True)
    path = os.path.join(upload_folder, filename)
    f.save(path)

    # Validate saved image can be read (basic check against malformed uploads)
    try:
        import cv2
        img = cv2.imread(path)
        if img is None:
            try:
                os.remove(path)
            except Exception:
                pass
            return jsonify({'ok': False, 'error': 'Uploaded file is not a valid image'}), 400
    except Exception:
        current_app.logger.exception('Failed to validate uploaded image')

    try:
        res = get_model_service().predict_image(path)
        return jsonify({'ok': True, 'result': res, 'file': filename})
    except Exception:
        current_app.logger.exception('Image prediction failed')
        return jsonify({'ok': False, 'error': 'Prediction failed'}), 500
