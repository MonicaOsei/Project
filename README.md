# Diabetes Prediction API

## Problem Description

Diabetes is a chronic disease affecting millions of people worldwide. Early detection is crucial for effective management and prevention of complications. Traditional diagnosis requires multiple clinical tests and expert evaluation, which can be time-consuming and inaccessible in some settings.

This project provides a **machine learning-based prediction API** that estimates the likelihood of a patient having diabetes using common health metrics such as glucose level, blood pressure, BMI, age, and other clinical features. 

The solution can be used by healthcare professionals, researchers, or developers to **quickly assess diabetes risk**, integrate it into health applications, or automate preliminary screenings.

## Solution Overview

This project uses a **Random Forest classifier** trained on a diabetes dataset. The trained model is serialized using `pickle` and served through a **Flask API**, which allows users to send patient features and receive predictions along with probabilities.

The API exposes the following routes:

- `/` : Home route to check if the API is running.
- `/predict` : POST route that takes patient features in JSON format and returns the predicted class and probability.

## Installation & Running the Project

### Prerequisites

- Python 3.11 or higher
- Git
- Docker (optional, for containerized deployment)

### Steps to Run Locally

   ```bash
# 1. Clone the repository
git clone https://github.com/MonicaOsei/Project.git
cd Project

# 2. Create and activate virtual environment
python -m venv project_env
project_env\Scripts\activate   # Windows
source project_env/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the API
python predict.py

# 5. Test API with curl
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"features\": [6,148,72,35,0,33.6,0.627,50]}"

# 6. Build Docker image
docker build -t project_app .

# 7. Run Docker container
docker run -p 8000:8000 project_app

# 8. Test API in Docker (same curl as above)
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"features\": [6,148,72,35,0,33.6,0.627,50]}"
```
## Importance
- Provides quick diabetes risk predictions.
- Reduces healthcare provider workload.
- Useful for screenings in low-resource settings.
- Can be integrated into apps for real-time monitoring.
