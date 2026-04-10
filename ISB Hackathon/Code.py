import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify

# ---------------------------------------------------
# 1. GENERATE REALISTIC SYNTHETIC FRAUD DATA
# ---------------------------------------------------

def generate_realistic_data(n=400):

    np.random.seed(42)

    # Normal transaction amounts
    amount_normal = np.random.normal(2000, 800, n)

    # Fraud transactions usually higher
    amount_fraud = np.random.normal(9000, 2500, int(n*0.25))

    # Merge with 25% fraud ratio
    amounts = np.concatenate([amount_normal, amount_fraud])

    # is_night – fraud often higher at night
    is_night = np.concatenate([
        np.random.binomial(1, 0.3, n),         # normal users
        np.random.binomial(1, 0.7, int(n*0.25))  # fraud cases
    ])

    # new_device – fraudsters use new devices
    new_device = np.concatenate([
        np.random.binomial(1, 0.2, n),
        np.random.binomial(1, 0.6, int(n*0.25))
    ])

    # vpn – fraud uses VPN heavily
    vpn = np.concatenate([
        np.random.binomial(1, 0.1, n),
        np.random.binomial(1, 0.7, int(n*0.25))
    ])

    # Fraud labels
    is_fraud = np.array([0]*n + [1]*int(n*0.25))

    df = pd.DataFrame({
        "amount": amounts,
        "is_night": is_night,
        "new_device": new_device,
        "vpn": vpn,
        "is_fraud": is_fraud
    })

    return df


# ---------------------------------------------------
# 2. TRAIN THE MODEL
# ---------------------------------------------------

def train_fraud_model():

    df = generate_realistic_data()

    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    # Save model
    with open("fraud_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained successfully!")
    print("Dataset size:", len(df))
    print("Fraud Rate:", df["is_fraud"].mean())

train_fraud_model()


# ---------------------------------------------------
# 3. FLASK APP FOR PREDICTION
# ---------------------------------------------------

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_fraud():
    data = request.get_json()

    amount = data["amount"]
    is_night = data["is_night"]
    new_device = data["new_device"]
    vpn = data["vpn"]

    # Load trained model
    with open("fraud_model.pkl", "rb") as f:
        model = pickle.load(f)

    input_data = np.array([[amount, is_night, new_device, vpn]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return jsonify({
        "fraud_prediction": int(prediction),
        "fraud_probability": float(probability)
    })


if __name__ == "__main__":
    app.run(debug=True)