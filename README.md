## About

This project consists of two parts:

- A **Prediction Pipeline** Python script that fetches order data from Firestore, processes it, trains machine learning models to predict next-day item quantities and order totals, and posts these predictions to a remote API endpoint.

- A **Flask API** that receives prediction data via POST requests, stores it temporarily in memory, and serves the stored data via GET requests. It also has a health check endpoint and supports cross-origin requests (CORS) for frontend integration.

---

## API Endpoints

- `POST /api/predictions`  
  Accepts JSON data containing prediction details (e.g., prediction type, date, item name, predicted value) and stores it.

- `GET /api/predictions`  
  Returns all stored predictions as JSON.

- `GET /`  
  Health check endpoint to verify the API is running; returns a simple status message.
