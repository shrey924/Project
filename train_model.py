import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load Dataset
credit_card_data = pd.read_csv(r"S:\Project\ML Project\creditcard.csv")

# Separate Legit and Fraud
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

# Under Sampling
legit_sample = legit.sample(n=492, random_state=42)

# Combine Data
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# Features and Target
X = new_dataset.drop(columns="Class", axis=1)
y = new_dataset["Class"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "credit_card_model.pkl")

print("Model saved successfully!")