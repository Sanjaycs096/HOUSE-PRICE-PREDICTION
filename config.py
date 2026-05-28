import os


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'change-me')
    UPLOAD_FOLDER = os.path.join('static', 'uploads')
    MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
    CACHE_TYPE = 'simple'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB uploads
    ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'gif'}
    ALLOWED_IMAGE_MIME_PREFIX = 'image/'
    # Session cookie security defaults (override in ProdConfig)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    SESSION_COOKIE_SECURE = False


class DevConfig(Config):
    DEBUG = True


class ProdConfig(Config):
    DEBUG = False
    # In production, require secure cookies (HTTPS)
    SESSION_COOKIE_SECURE = True
