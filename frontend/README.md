# Frontend for Action Recommendation

This folder contains a simple web frontend that uses the same recommendation logic as `four.ipynb`.

## What it does
- Loads `customer_cleaned.csv`
- Trains the delay-risk model (same threshold + RandomForest approach as notebook)
- Accepts order feature inputs from a web form
- Returns recommended action, priority, owner, and SLA

## Run locally
1. Open terminal in this folder:
   cd frontend
2. Install dependencies:
   pip install -r requirements.txt
3. Start the app:
   python app.py
4. Open:
   http://127.0.0.1:5000

## Optional API usage
POST JSON to:
- `/api/recommend`

Example JSON body (keys should match feature names from your dataset):
```json
{
  "Type": "DEBIT",
  "Days for shipping (real)": 3,
  "Days for shipment (scheduled)": 2
}
```
