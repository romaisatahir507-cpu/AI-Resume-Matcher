import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#--------------------------
# 1. Load Training Dataset
#--------------------------

data = pd.read_csv("ml/training_data.csv")

print("Training data loaded successfully.")
print(f"Number of Records: {len(data)}")


#-------------------------
# 2. define Features
#-------------------------

features = [
    "tfidf_score",
    "semantic_score",
    "skills_match",
    "experience_match",
    "education_match",
    "missing_skills_ratio"
]

X = data[features]

y = data["match_score"]


#-------------------------
# 3. Split Dataset
#-------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")


#--------------------------
# 4. Create Random Forest Model
#--------------------------

model = RandomForestRegressor(
    n_estimators = 100,
    random_state = 42
)


#--------------------------
# 5. Train Model
#--------------------------

model.fit(X_train, y_train)

print("Model training completed.")


#--------------------------
# 6. Make Predictions 
#--------------------------

predictions = model.predict(X_test)


#--------------------------
# 7. Evaluate Model
#--------------------------

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


print("\nModel Evaluation:")
print("--------------------------")
print(f"MAE : {mae: .2f}")
print(f"MSE : {mse: .2f}")
print(f"R Square : {r2: .2f}")


# ---------------------------------------
# 8. Feature Importance
# ---------------------------------------

print("\nFeature Importance:")
print("--------------------------")

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

for _, row in feature_importance.iterrows():

    print(
        f"{row['Feature']:<25} "
        f"{row['Importance']:.4f}"
    )



#----------------------------
# 9. Train Final Model
#----------------------------

final_model = RandomForestRegressor(
    n_estimators = 100,
    random_state = 42
)

final_model.fit(X, y)


#----------------------------
# 10. Save Final Model
#----------------------------

joblib.dump(
    final_model,
    "ml/resume_match_model.pkl"
)

print("\nFinal Model trained successfully.")

print(
    "Model saved to: "
    "ml/resume_match_model.pkl"
)

