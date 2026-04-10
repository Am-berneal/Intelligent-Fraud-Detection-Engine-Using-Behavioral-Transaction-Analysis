# 🛡️ Fraud Detection System using Machine Learning & Flask API

This project is developed for the ISB Hackathon and demonstrates an AI-based fraud detection system using Machine Learning and Flask API for real-time predictions.

## 🚀 Overview
The system detects fraudulent transactions based on behavioral patterns such as transaction amount, time of activity, device usage, and VPN usage. It is trained using a synthetic dataset designed to simulate real-world fraud scenarios.

## 🧠 Approach

### 1. Synthetic Data Generation
A realistic dataset is created with features:
- Transaction amount
- Night-time activity
- New device usage
- VPN usage

Fraud patterns are simulated based on higher risk behaviors.

### 2. Machine Learning Model
- Algorithm: Random Forest Classifier
- Library: Scikit-learn
- Model trained and saved using Pickle

### 3. Flask API
A REST API is built using Flask to provide real-time fraud predictions.

**Endpoint:**
POST `/predict`

## 📥 Input Example
```json
{
  "amount": 5000,
  "is_night": 1,
  "new_device": 0,
  "vpn": 1
}
