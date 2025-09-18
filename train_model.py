import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv("crop_dataset.csv")
print("Dataset loaded:", df.shape)

# 2. Initialize label encoders
district_encoder = LabelEncoder()
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()

# 3. Encode categorical features
df["District_Name_enc"] = district_encoder.fit_transform(df["District_Name"])
df["Soil_color_enc"] = soil_encoder.fit_transform(df["Soil_color"])
df["Crop_enc"] = crop_encoder.fit_transform(df["Crop"])

# 4. Define features and target
X = df[["District_Name_enc", "Soil_color_enc",
        "Nitrogen", "Phosphorus", "Potassium",
        "pH", "Rainfall", "Temperature"]]

y = df["Crop_enc"]

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with accuracy: {acc:.2f}")

# 8. Save model and encoders
joblib.dump(model, "crop_model.pkl")
joblib.dump(district_encoder, "district_encoder.pkl")
joblib.dump(soil_encoder, "soil_encoder.pkl")
joblib.dump(crop_encoder, "crop_encoder.pkl")

print("✅ Model and encoders saved!")
