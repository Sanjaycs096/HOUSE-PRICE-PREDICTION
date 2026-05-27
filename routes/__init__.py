from flask import Blueprint

def register_blueprints(app):
    from .main import main_bp
    from .predict import predict_bp
    from .market import market_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(predict_bp, url_prefix='/api')
    app.register_blueprint(market_bp, url_prefix='/api')
