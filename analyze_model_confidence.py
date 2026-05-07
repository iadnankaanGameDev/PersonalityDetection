"""
Manual model confidence analysis script.

This script loads the saved personality classification model package
and tests how the Random Forest vote distribution behaves when all
questionnaire answers are set to the same value.

It is used only for analysis/debugging and is not required to run the FastAPI app.
"""


import pickle
import pandas as pd

with open("personality_model_package.pkl", "rb") as f:
    model_package = pickle.load(f)

model = model_package["model"]
features = model_package["features"]

print("Model loaded successfully")
print("Number of features:", len(features))
print("Classes:", model.classes_)

def show_prediction_distribution(input_value: int, title: str):
    sample = pd.DataFrame([{feature: input_value for feature in features}])
    sample = sample[features]

    proba = model.predict_proba(sample)[0]

    proba_df = (
        pd.DataFrame({
            "Personality": model.classes_,
            "Probability": proba
        })
        .sort_values(by="Probability", ascending=False)
    )

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(proba_df)

show_prediction_distribution(0, "All Neutral")
show_prediction_distribution(1, "All Slightly Agree")
show_prediction_distribution(2, "All Agree")
show_prediction_distribution(3, "All Strongly Agree")
show_prediction_distribution(-2, "All Disagree")

for value, title in [
    (0, "All Neutral"),
    (1, "All Slightly Agree"),
    (2, "All Agree"),
    (3, "All Strongly Agree"),
    (-2, "All Disagree"),
]:
    sample = pd.DataFrame([{feature: value for feature in features}])
    sample = sample[features]
    max_probability = model.predict_proba(sample)[0].max()
    prediction = model.predict(sample)[0]

    print(f"{title}: Prediction={prediction}, Max Probability={max_probability:.2%}")