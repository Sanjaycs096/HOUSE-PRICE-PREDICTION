# 🏠 House Price Prediction (AI-Powered)

An AI-powered web application that estimates property prices based on various input features and images.

This tool helps real estate agents and home buyers get instant, data-driven price estimations to make informed property decisions.

[Live Demo](#demo) · [Repository](https://github.com/Sanjaycs096/HOUSE-PRICE-PREDICTION) · [Report Bug](https://github.com/Sanjaycs096/HOUSE-PRICE-PREDICTION/issues) · [Request Feature](https://github.com/Sanjaycs096/HOUSE-PRICE-PREDICTION/issues)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Demo](#demo)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Environment Variables](#environment-variables)
- [Installation](#installation)
- [Running Locally](#running-locally)
- [Testing](#testing)
- [Security](#security)
- [Deployment](#deployment)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## Overview

The House Price Prediction application leverages machine learning models to provide accurate real estate valuations. Users can input standard property metrics (area, BHK, location, etc.) or upload property images. The application parses these inputs through Scikit-learn and ONNX models to return a calculated price estimate.

## Key Features

- **Interactive Web UI** – Clean and modern design for smooth user experience.
- **Form-Based Prediction** – Enter property details to get instant price estimation.
- **Image Upload Support** – Upload an image of the property to assist prediction via an ONNX model.
- **AI/ML Model Integration** – Backend models trained on comprehensive real estate datasets.
- **Responsive Design** – Fully functional across desktops, tablets, and mobile devices.

## Demo

Live demo is not currently available.

## Architecture

The application follows a monolithic client-server architecture built on Flask. The frontend uses server-side rendered Jinja2 templates (HTML/CSS/JS). The backend routing handles form submissions, normalizes data, and passes inputs to a `ModelService` which interfaces with pre-trained machine learning models (.pkl for tabular data, .onnx for image data).

## Tech Stack

**Frontend:** HTML, CSS, JavaScript (Vanilla)
**Backend:** Python, Flask
**Machine Learning:** Scikit-learn, ONNX Runtime, Pandas, NumPy, Pillow
**Testing:** Pytest, Flake8
**Infrastructure/Deployment:** Vercel (Serverless Functions)

## Project Structure

```text
├── .github/             # GitHub configuration and actions
├── api/                 # Vercel serverless entrypoint
├── models/              # Pre-trained ML models (.pkl, .onnx, .h5)
├── routes/              # Flask application routing
├── services/            # Business logic and model integration
├── static/              # CSS, JS, and uploads
├── templates/           # Jinja2 HTML templates
├── tests/               # Pytest suite
├── app.py               # Flask application factory
├── config.py            # Environment configuration
└── vercel.json          # Vercel deployment config
```

## Prerequisites

- Python 3.12+
- Git

## Environment Variables

Copy the `.env.example` file to `.env` and configure the following variables:

- `SECRET_KEY`: Used for Flask session security and CSRF protection.
- `VERCEL`: Set to `False` for local development.

## Installation

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
   *For local image-model support on Windows, also run:*
   ```bash
   pip install -r requirements-local.txt
   ```

## Running Locally

Run the Flask development server:

```bash
python app.py
```
Then open `http://127.0.0.1:5000` in your web browser.

## Testing

To run the test suite and linters:

```bash
pip install -r requirements-test.txt
pytest tests/
flake8 .
```

## Security

- **Rate Limiting**: Integrated `Flask-Limiter` to prevent brute-force and DDoS attacks on prediction endpoints.
- **CSP**: Content Security Policy configured via `Flask-Talisman` to prevent XSS.
- **CSRF Protection**: Manual CSRF token generation and validation for form submissions.
- **File Validation**: Image uploads are strictly validated by extension and MIME type.

## Deployment

This repository is configured for serverless deployment on **Vercel**.

1. Install the Vercel CLI: `npm i -g vercel`
2. Login: `vercel login`
3. Deploy: `vercel --prod`

*Note: Large model files (like HDF5) may exceed Vercel's serverless size limits. The application defaults to using the optimized ONNX runtime for image predictions in production.*

## Known Limitations

- **Image Verification**: If a non-house image (such as an image of a tree, car, or unrelated object) is uploaded, the model will still predict a house price because it is not trained to detect whether the image is actually a house.

## Roadmap

- Add image classification to verify if the uploaded image is actually a house before price prediction.
- Improve dataset quality with more diverse property images.
- Integrate live property market APIs for dynamic pricing.
- Migrate to a dedicated model-serving architecture for heavier models.

## Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Author

**Sanjay** - [Sanjaycs096](https://github.com/Sanjaycs096)
