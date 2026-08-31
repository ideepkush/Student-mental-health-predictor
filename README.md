# Student Mental Health Score Predictor

Predicts a student's self-reported mental health score (0 to 10) from their social media habits, sleep, study time, and activity levels.

**[Live demo](https://student-mental-health-predictor-wrfz.onrender.com/ui)** | [Notebook](Mental_health.ipynb) | [API docs](https://student-mental-health-predictor-wrfz.onrender.com/docs)

---

## Summary

- **Problem:** student support teams have no cheap, early signal for who might be struggling until a student asks for help.
- **Approach:** a tuned **KNN** regression model on **5,000** survey records, served as a scikit-learn pipeline behind a FastAPI endpoint.
- **Result:** **R² 0.891** with a mean absolute error of **0.31** points on the 0 to 10 scale, beating a linear baseline (R² 0.721).
- **Live:** deployed on Render with an interactive form and a REST API. Also runs locally with one command.

---

## Why this matters

- Universities and counselling services want an early, low-cost way to spot students who may need support before it turns into a crisis.
- Every input the model uses (sleep, screen time, study balance, activity) is something a student can change, so it also works as a "what if" tool.
- Scope: this is an analysis of self-reported survey data, not a clinical or diagnostic tool.

---

## Results

**Dataset:** Kaggle "Student Social Media and Mental Health Impact", **5,000 records**, 13 features covering demographics, social media habits, and lifestyle. Target is a continuous wellbeing score from 0 to 10.

Model comparison with 5-fold cross-validation:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression (baseline) | 0.523 | 0.664 | 0.721 |
| Gradient Boosting | 0.449 | 0.580 | 0.787 |
| SVR | 0.433 | 0.579 | 0.788 |
| Random Forest | 0.371 | 0.503 | 0.840 |
| **KNN** | **0.341** | **0.484** | **0.851** |

Final tuned KNN on the held-out test set:

- MAE: **0.309**
- RMSE: **0.437**
- R²: **0.891**, up **0.17** from the linear baseline

Strongest signals in the data:

| Feature | Correlation with score |
|---|---|
| Average daily social media use | **-0.82** |
| Daily phone unlocks | **-0.79** |
| Sleep hours | **+0.77** |
| Study hours | **+0.75** |
| Physical activity | **+0.52** |

---

## Key decisions

- Chose KNN over Random Forest, SVR, and linear models because it had the best cross-validated R² (0.851) and held up after tuning.
- Dropped the daily phone-unlocks feature despite its strong correlation, because its VIF was above 12 and it mostly duplicated daily usage hours.
- Collapsed 111 countries into the top 10 plus an "Other" group to avoid a long tail of near-empty one-hot columns.
- Reported MAE and RMSE next to R², because on a 0 to 10 scale a single R² is easy to misread. MAE 0.31 means predictions land within about a third of a point.
- Kept the framing as survey analysis, not diagnosis, and evaluated it on that basis.

---

## How it works

Raw data, then EDA, cleaning, feature engineering, preprocessing, model comparison, tuning, and deployment.

- Cleaning: remove duplicates, clip negative physical-activity values.
- Feature engineering: group rare countries, drop the collinear unlocks feature.
- Preprocessing: `StandardScaler` for numeric features, `OrdinalEncoder` for stress level, `OneHotEncoder` for the rest.
- Tuning: `GridSearchCV` over the KNN hyperparameters.
- Serving: a single `/predict` endpoint returns the score and a short interpretation band.

---

## Tech stack

| Layer | Tools |
|---|---|
| ML and data | Python, scikit-learn, pandas, numpy |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | HTML, CSS, vanilla JS |
| Deployment | Docker, Render |

---

## Run locally

```bash
git clone https://github.com/ideepkush/Student-mental-health-predictor.git
cd Student-mental-health-predictor
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
# open http://127.0.0.1:8000/ui
```

Docker:

```bash
docker build -t mental-health-predictor .
docker run -p 8000:8000 mental-health-predictor
```

---

## API

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict` | Predict mental health score from student inputs |
| GET | `/ui` | Interactive form |
| GET | `/docs` | Swagger UI |
| GET | `/health` | Health check |

<details>
<summary>Sample request and response</summary>

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 21, "Gender": "Male", "Country": "India",
    "Academic_Level": "Undergraduate",
    "Most_Used_Platform": "Instagram",
    "Purpose_Of_Use": "Entertainment",
    "Avg_Daily_Usage_Hours": 4.0, "Study_Hours": 3.0,
    "Physical_Activity_Hours": 1.5, "Sleep_Hours_Per_Night": 7.0,
    "Stress_Level": "Low"
  }'
```

```json
{
  "mental_health_score": 6.639,
  "interpretation": "Good mental health"
}
```
</details>

---

## Author

**Deepak Kushwaha**, Data Scientist, MSc at University of Naples Federico II
[GitHub](https://github.com/ideepkush) | [LinkedIn](https://www.linkedin.com/in/deepak-kushwaha-75155013a/)
