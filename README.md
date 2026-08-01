# Student Social Media & Mental Health Score Predictor

An end-to-end machine learning web application that predicts a student's mental health score based on social media usage habits, lifestyle, and demographic data.

---

## Demo

> 🚀 **Live Demo:** [https://student-mental-health-predictor-wrfz.onrender.com/ui](https://student-mental-health-predictor-wrfz.onrender.com/ui)

![UI Preview](https://img.shields.io/badge/Frontend-Interactive%20Form-6366f1?style=flat-square)
![API](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square)
![Model](https://img.shields.io/badge/Model-KNN%20%7C%20R²%200.891-22c55e?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11-3b82f6?style=flat-square)
![Deployed on Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=flat-square&logo=render)

---

## Project Overview

This project explores how social media usage, sleep, study hours, physical activity, and stress levels impact student mental health. A regression model is trained to predict a continuous **Mental Health Score (0–10)**.

### Dataset
- **Source:** [Kaggle — Student Social Media and Mental Health Impact](https://www.kaggle.com/datasets/shivasingh4945/student-social-media-and-mental-health-impact)
- **Size:** 5,000 student records
- **Features:** 13 columns (demographics, social media habits, lifestyle)

---

## Key Findings

| Feature | Correlation with Mental Health Score |
|---|---|
| Avg Daily Social Media Usage | **-0.82** (strong negative) |
| Daily Phone Unlocks | -0.79 (strong negative) |
| Sleep Hours | **+0.77** (strong positive) |
| Study Hours | +0.75 (strong positive) |
| Physical Activity | +0.52 (moderate positive) |
| Stress Level | Strong negative trend |

---

## ML Pipeline

```
Raw Data → EDA → Cleaning → Feature Engineering → Preprocessing → Model Selection → Hyperparameter Tuning → Deployment
```

### Steps
1. **EDA** — distribution analysis, correlation heatmap, bivariate plots
2. **Cleaning** — removed duplicates, clipped negative physical activity values
3. **Feature Engineering** — grouped 111 countries → top 10 + Other, dropped `Daily_Unlocks` (VIF > 12)
4. **Preprocessing** — StandardScaler (numeric), OrdinalEncoder (stress level), OneHotEncoder (categorical)
5. **Model Comparison** — 9 models evaluated via 5-fold cross-validation

### Model Comparison Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **KNN** | 0.341 | 0.484 | **0.851** |
| Random Forest | 0.371 | 0.503 | 0.840 |
| SVR | 0.433 | 0.579 | 0.788 |
| Gradient Boosting | 0.449 | 0.580 | 0.787 |
| Linear Regression | 0.523 | 0.664 | 0.721 |

### Final Model — Tuned KNN

| Metric | Score |
|---|---|
| MAE | 0.309 |
| RMSE | 0.437 |
| **R²** | **0.891** |

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML / Data | Python, scikit-learn, pandas, numpy |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | HTML, CSS, JavaScript (no framework) |
| Deployment | Docker, Render |

---

## Project Structure

```
├── app.py                  # FastAPI application
├── index.html              # Interactive frontend
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container setup
├── .dockerignore
├── models/
│   └── knn_mental_health.pkl   # Trained model pipeline
└── Mental_health.ipynb     # Full analysis notebook
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/ui` | Frontend |
| GET | `/docs` | Swagger UI |
| POST | `/predict` | Predict mental health score |

### Sample Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 21,
    "Gender": "Male",
    "Country": "India",
    "Academic_Level": "Undergraduate",
    "Most_Used_Platform": "Instagram",
    "Purpose_Of_Use": "Entertainment",
    "Avg_Daily_Usage_Hours": 4.0,
    "Study_Hours": 3.0,
    "Physical_Activity_Hours": 1.5,
    "Sleep_Hours_Per_Night": 7.0,
    "Stress_Level": "Low"
  }'
```

### Sample Response

```json
{
  "mental_health_score": 6.639,
  "interpretation": "Good mental health"
}
```

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/ideepkush/Student-mental-health-predictor.git
cd Student-mental-health-predictor

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn app:app --reload

# 5. Open in browser
# http://127.0.0.1:8000/ui
```

## Run with Docker

```bash
docker build -t mental-health-predictor .
docker run -p 8000:8000 mental-health-predictor
```

---

## Author

**Deepak Kushwaha**  
Data Scientist | MSc — University of Naples Federico II  
[GitHub](https://github.com/ideepkush) · [LinkedIn](https://www.linkedin.com/in/deepak-kushwaha-75155013a/)
