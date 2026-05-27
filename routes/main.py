from flask import Blueprint, render_template

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    return render_template('index.html')


@main_bp.route('/predict')
def predict():
    return render_template('predict.html')


@main_bp.route('/features')
def features():
    return render_template('features.html')
