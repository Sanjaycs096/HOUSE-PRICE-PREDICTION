from flask import Flask, session
from config import DevConfig
from routes import register_blueprints
from extensions import limiter, talisman
import os
import secrets




def create_app(config_object=DevConfig):
    app = Flask(__name__, static_folder='static', template_folder='templates')
    app.config.from_object(config_object)

    # Vercel uses a read-only filesystem for the deployed app bundle.
    # Use /tmp for any runtime file writes there, but keep the local path for development.
    upload_folder = app.config.get('UPLOAD_FOLDER', 'static/uploads')
    if os.environ.get('VERCEL'):
        upload_folder = os.path.join('/tmp', 'uploads')
    app.config['UPLOAD_FOLDER'] = upload_folder
    os.makedirs(upload_folder, exist_ok=True)

    # Initialize security extensions
    limiter.init_app(app)
    # Basic CSP - allow self resources and data: for images
    csp = {
        'default-src': ["'self'"],
        'img-src': ["'self'", 'data:'],
        'script-src': ["'self'", "'unsafe-inline'", 'https://cdn.jsdelivr.net', 'https://unpkg.com', 'https://cdnjs.cloudflare.com'],
        'style-src': ["'self'", "'unsafe-inline'", 'https://fonts.googleapis.com', 'https://cdnjs.cloudflare.com'],
        'connect-src': ["'self'", 'https://cdn.jsdelivr.net', 'https://unpkg.com', 'https://cdnjs.cloudflare.com'],
        'font-src': ['https://fonts.gstatic.com', 'https://fonts.googleapis.com']
    }
    # Don't force HTTPS redirects in debug mode (use HTTPS in production)
    talisman.init_app(app, content_security_policy=csp, force_https=not app.debug)

    # Ensure a CSRF token exists in session
    @app.before_request
    def _ensure_csrf():
        if 'csrf_token' not in session:
            session['csrf_token'] = secrets.token_urlsafe(32)

    @app.context_processor
    def inject_csrf_token():
        return {'csrf_token': session.get('csrf_token', '')}

    # Register blueprints after extensions
    register_blueprints(app)

    # Generic error handler for production to avoid leaking traces
    @app.errorhandler(Exception)
    def handle_exception(e):
        app.logger.exception('Unhandled exception')
        return ('Internal server error', 500)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
