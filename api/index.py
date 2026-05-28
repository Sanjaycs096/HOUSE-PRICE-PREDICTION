from app import create_app
from vercel_wsgi import make_handler

# Create Flask app and expose a Vercel-compatible handler
app = create_app()
handler = make_handler(app)
