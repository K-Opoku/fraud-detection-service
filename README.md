# End-to-End Fraud Detection Service

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.121.1-05998b)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.1.1-blue)](https://xgboost.ai/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This project is a complete, production-ready machine learning service for real-time fraud detection. It's built as an end-to-end solution, from data cleaning and model training to a containerized REST API service.

The final artifact is a **Docker image** that runs a **FastAPI** application, which serves a trained **XGBoost** model to make on-demand predictions.

## 1. 🎯 Business Goal & Project Objective

The most critical part of this project was defining a clear business objective.

**Problem:** Financial fraud is highly imbalanced. Missing a single fraudulent transaction (a **False Negative**) is often far more costly than flagging a legitimate transaction for review (a **False Positive**).

**Project Objective:** Prioritize **High Recall** to minimize missed fraud, while maintaining acceptable precision.

* **Primary Goal: `Recall >= 85%`** (We must catch at least 85% of all fraud).
* **Secondary Goal: `Precision >= 40%`** (At least 40% of our alerts must be correct).

## 2. 📈 Final Model Performance

After experimenting with Logistic Regression, Random Forest, and XGBoost, the **XGBoost Classifier** was selected for its superior performance. A custom probability threshold of `0.038` was determined (instead of the default 0.5) to achieve our business goal.

**Result: Objective Met.**

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Recall** | **`0.888`** | **Goal Met.** We successfully identify **88.8%** of all fraudulent transactions. |
| **Precision** | **`0.500`** | **Goal Met.** When we flag a transaction as fraud, it is correct 50% of the time. |
| **ROC AUC Score** | **`0.998`** | Demonstrates the model has outstanding separability between classes. |
| **Avg. Precision (PR-AUC)** | `0.445` | The area under the Precision-Recall curve, indicating robust performance. |

## 3. 🛠️ Tech Stack & Architecture

This project uses a modern, high-performance tech stack ideal for ML production environments.

* **Python Version:** `3.12`
* **ML Model:** `XGBoost`
* **ML Pipeline:** `Scikit-Learn` (for preprocessing)
* **API Framework:** `FastAPI` (for high-speed asynchronous serving)
* **Data Validation:** `Pydantic` (for robust API request/response schemas)
* **Containerization:** `Docker`
* **Dependency Management:** `uv`

### Service Architecture
The architecture is a simple, robust, and scalable microservice.

`Client Request (JSON)` → `Docker Container` → `FastAPI / Uvicorn` → `Pydantic Data Validation` → `Scikit-Learn Pipeline (Prediction)` → `JSON Response`

## 4. 🔬 ML Pipeline Deep Dive

The final artifact, `final_model_pipeline.pkl`, is not just a model but a full `Scikit-Learn` pipeline that handles all preprocessing. This ensures that the raw data sent to the API is transformed in the exact same way as the training data.

1.  **Input Features (7 total):**
    * `step`, `type`, `amount`, `oldbalanceorg`, `newbalanceorig`, `oldbalancedest`, `newbalancedest`

2.  **Preprocessing (`ColumnTransformer`):**
    * **Log Transformation:** Applies `np.log1p` to 5 skewed numerical features: `amount`, `oldbalanceorg`, `newbalanceorig`, `oldbalancedest`, `newbalancedest`.
    * **One-Hot Encoding:** Applies `OneHotEncoder` to the `type` categorical feature.
    * **Passthrough:** The `step` feature is used as-is.

3.  **Model Training (`XGBClassifier`):**
    * To handle the extreme class imbalance, the model was trained with `scale_pos_weight=10`, which gives a 10x higher penalty to misclassifying a positive (fraud) case.

## 5. 🚀 How to Run This Service

This service is fully containerized with Docker.

### Prerequisites
* [Docker](https://www.docker.com/get-started) installed.
* `git` installed.

### Step 1: Clone the Repository
Open the terminal and run:
```bash
git clone https://github.com/K-Opoku/fraud-detection-service.git

cd fraud-detection-service
```
Or visit the repo:
[https://github.com/K-Opoku/fraud-detection-service.git](https://github.com/K-Opoku/fraud-detection-service.git)

### Step 2: Build the Docker Image

This will build the image, installing all dependencies from `pyproject.toml` and `uv.lock`.

```bash
docker build -t fraud-detection-service .
```

### Step 3: Run the Docker Container

This command runs the container and maps your local port `8000` to the container's port `8000`.

```bash
docker run -p 8000:8000 fraud-detection-service
```
The API is now running and accessible at `http://localhost:8000`.

## 6. 📨 API Endpoints

The service provides one main endpoint for prediction.

### `POST /predict`

This endpoint accepts a single transaction as a JSON object and returns its fraud probability and a final decision.

#### Request Schema (Enforced by Pydantic)
The API will *only* accept data matching this schema. Note that `type` must be one of the 5 specified strings.

```python
class Transaction(BaseModel):
    step: int
    type: Literal['cash_in', 'cash_out', 'debit', 'payment', 'transfer']
    amount: float = Field(..., ge=0.0)
    oldbalanceorg: float = Field(..., ge=0.0)
    newbalanceorig: float = Field(..., ge=0.0)
    oldbalancedest: float = Field(..., ge=0.0)
    newbalancedest: float = Field(..., ge=0.0)
```

#### Example `curl` Request
(You can run this from a new terminal while the service is running).

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "step": 1,
           "type": "cash_in",
           "amount": 57809.81,
           "oldbalanceorg": 3434765.61,
           "newbalanceorig": 3492575.42,
           "oldbalancedest": 59666.0,
           "newbalancedest": 0.0
         }'
```
#### You can also send a request using the test_api .py in the test folder
(you can run it in the terminal like this)
```bash
python tests/test_api.py
```

#### Example Success Response
The API returns the probability and the final boolean decision (based on the `0.038` threshold).

```json
{
  "fraud_probability": 0.002154,
  "fraud": false
}
```

---

## 7. 💡 Future Improvements

* **CI/CD Pipeline:** Implement GitHub Actions to automatically test and build the Docker image on every push to `main` and publish it to Docker Hub or GHCR.
* **Cloud Deployment:** Deploy the service to a scalable cloud platform like **AWS Elastic Beanstalk**, **GCP Cloud Run**, or a **Kubernetes** cluster.
* **Model Monitoring:** Integrate tools like **Evidently AI** or **Prometheus** to monitor the API for latency, traffic, and model drift over time.
* **Unit & Integration Testing:** Expand `test_api.py` into a full `pytest` suite to test API endpoints, data validation, and model edge cases.
