# Personality Detection

An educational multi-class machine learning web app that predicts one of 16 personality classes based on questionnaire-style responses.

The project uses a **Random Forest Classifier** trained on a synthetic personality dataset and is deployed locally with **FastAPI** and a custom dark-themed web interface.

> Important: This project is for educational machine learning practice only.  
> The dataset is synthetic, and the predictions should not be interpreted as real psychological assessment results.

---

## Project Overview

This project was built as an end-to-end machine learning deployment practice project.

The main goals were:

- Build a multi-class classification model
- Compare different classification algorithms
- Evaluate model performance using appropriate metrics
- Save the trained model with Pickle
- Build a FastAPI web application
- Create a polished bilingual UI with English/Turkish support
- Display prediction confidence and top alternative predictions

---

## Dataset

The dataset contains questionnaire-style answers for 16 personality classes.

Target column:

- `Personality`

Input features:

- 60 questionnaire response columns
- Each response is represented numerically from `-3` to `3`

Scale:

| Value | Meaning |
|---:|---|
| -3 | Strongly Disagree |
| -2 | Disagree |
| -1 | Slightly Disagree |
| 0 | Neutral |
| 1 | Slightly Agree |
| 2 | Agree |
| 3 | Strongly Agree |

The dataset is synthetic, so model results should be understood as pattern recognition within generated data rather than real-world personality prediction reliability.

---

## Personality Classes

The model predicts one of the following 16 classes:

| Code | Role |
|---|---|
| INTJ | Architect |
| INTP | Logician |
| ENTJ | Commander |
| ENTP | Debater |
| INFJ | Advocate |
| INFP | Mediator |
| ENFJ | Protagonist |
| ENFP | Campaigner |
| ISTJ | Logistician |
| ISFJ | Defender |
| ESTJ | Executive |
| ESFJ | Consul |
| ISTP | Virtuoso |
| ISFP | Adventurer |
| ESTP | Entrepreneur |
| ESFP | Entertainer |

---

## Model Development

Several models were tested and compared:

- Logistic Regression Baseline
- Tuned Logistic Regression
- Tuned Linear SVM
- Tuned Random Forest
- Controlled Random Forest

Final comparison:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Random Forest Tuned | 0.9806 | 0.9806 | 0.9806 | 0.9806 |
| Random Forest Controlled | 0.9754 | 0.9754 | 0.9754 | 0.9754 |
| Logistic Regression Tuned | 0.9188 | 0.9187 | 0.9187 | 0.9187 |
| Logistic Regression Baseline | 0.9183 | 0.9183 | 0.9183 | 0.9182 |
| Linear SVM Tuned | 0.9103 | 0.9102 | 0.9103 | 0.9101 |

The tuned Random Forest achieved the best test performance. However, since it reached perfect training performance, mild overfitting was observed. The final interpretation is cautious because the dataset is synthetic.

---

## Prediction Confidence

The app displays:

- Predicted personality class
- Model vote confidence
- Top 3 alternative predictions
- A confidence note

For Random Forest, the displayed confidence is based on the model's class vote distribution, not psychological certainty.

Manual confidence checks showed that uniform answer patterns, such as answering all questions with the same value, produce low confidence. This suggests that the model relies on more distinctive response patterns rather than simply high or low agreement values.

---

## Web App Features

The FastAPI app includes:

- Dark premium UI
- English/Turkish language toggle
- 60-question input form
- Preserved selected answers after prediction
- Reset Answers button
- Prediction result card
- Top 3 predictions
- Model confidence note
- Personality type infographic

---

## Project Structure

```text
PersonalityDetection/
│
├── app.py
├── analyze_model_confidence.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── static/
│   └── images/
│       └── personalities.png
│
└── templates/
    └── index.html

---

## Model File Notice

The trained model file is not included in this repository because it is too large for normal GitHub storage.

Ignored model file:

```text
personality_model_package.pkl
```

To run the app locally, place the trained model package in the project root:

```text
PersonalityDetection/
├── personality_model_package.pkl
├── app.py
└── ...
```

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.\.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Locally

Start the FastAPI server:

```bash
uvicorn app:app
```

Then open:

```text
http://127.0.0.1:8000
```

For development, reload mode can also be used:

```bash
uvicorn app:app --reload
```

> Note: If the model file is large, running without `--reload` may be more stable.

---

## Confidence Analysis Script

The repository includes:

```text
analyze_model_confidence.py
```

This script loads the saved model package and checks how the model's prediction confidence behaves under simple manual input patterns, such as:

- All Neutral
- All Slightly Agree
- All Agree
- All Strongly Agree
- All Disagree

Run it with:

```bash
python analyze_model_confidence.py
```

This script is for analysis/debugging only and is not required to run the web app.

---

## Technologies Used

- Python
- pandas
- scikit-learn
- FastAPI
- Jinja2
- Uvicorn
- HTML/CSS
- Git/GitHub

---

## Disclaimer

This app is an educational machine learning project.

It should not be used as:

- a real psychological assessment tool
- a clinical or professional personality test
- a decision-making tool for hiring, diagnosis, counseling, or evaluation

The dataset is synthetic and the model predictions are only demonstrations of machine learning classification behavior.