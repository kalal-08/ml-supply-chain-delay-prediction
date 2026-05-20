# Supply Chain Delay Risk & Action Recommendation

A concise end-to-end AIML pipeline for supply-chain order delay prediction, risk scoring, and operational action recommendations.

## Project Overview
- Input data: `customer.csv`
- Cleaned/model-ready data: `customer_cleaned.csv`
- Risk scoring output: `risk_scored_orders.csv`
- Action recommendation output: `action_recommendations.csv`
- Notebooks:
  - `one.ipynb`: cleaning + preprocessing
  - `two.ipynb`: baseline modeling (regression + classification)
  - `three.ipynb`: risk scoring
  - `four.ipynb`: action recommendation
  - `five.ipynb`: visualization/dashboard-style analysis
- Web app: `frontend/app.py`

## Key Data Facts
- `customer.csv`: 751 records (plus header)
- `customer_cleaned.csv`: 750 records (plus header)
- Cleaning summary from notebook:
  - Initial shape: `(750, 12)`
  - Processed shape: `(750, 26)`
  - Duplicate rows removed: `0`
  - Dropped ID columns: `customer_id`, `order_id`
  - Date features engineered; raw date columns dropped

## Modeling Results
### 1) Lead Time Regression (`two.ipynb`)
- Best shown model: `RandomForestRegressor`
- `MAE`: `0.236711`
- `MSE`: `0.246481`
- `R2`: `0.929357`

### 2) Delay Classification (`two.ipynb`)
- Delay label threshold: `lead_time_days > 16.0` (75th percentile)
- Features used: `25`
- Accuracy: `1.00`
- Confusion matrix:
  - `[[231, 0], [0, 32]]`
- Support: `263`

### 3) Risk Scoring Model (`three.ipynb`)
- Delay threshold: `16.0`
- Delay class proportion:
  - Class `0`: `0.877333`
  - Class `1`: `0.122667`
- Accuracy: `1.00`
- Confusion matrix:
  - `[[132, 0], [0, 18]]`
- Support: `150`

## Risk & Action Outputs
### `risk_scored_orders.csv`
- Rows: `750`
- Important output fields:
  - `Delay_Probability`
  - `Risk_Level`
  - `Recommended_Action`
- Risk distribution:
  - `Low`: `658`
  - `High`: `90`
  - `Medium`: `2`

### `action_recommendations.csv`
- Rows: `750`
- Important output fields:
  - `Delay_Probability`, `Risk_Level`
  - `Action`, `Priority`, `Owner`, `SLA_Hours`
- Risk distribution:
  - `Low`: `658`
  - `High`: `90`
  - `Medium`: `2`
- Priority distribution:
  - `P3`: `658`
  - `P1`: `90`
  - `P2`: `2`

## Action Logic (from `four.ipynb`)
- `High` risk:
  - Action: Expedite shipment and escalate supplier
  - Priority: `P1`
  - SLA: `12` hours
- `Medium` risk:
  - Action: Monitor order and confirm dispatch timeline
  - Priority: `P2`
  - SLA: `24` hours
- `Low` risk:
  - Action: No immediate action; keep routine monitoring
  - Priority: `P3`
  - SLA: `72` hours

## Run the Frontend
```bash
cd frontend
pip install -r requirements.txt
python app.py
```
Open: `http://127.0.0.1:5000`

## Note
- Reported classification accuracy is perfect (`1.00`) in notebook splits. Validate on a truly unseen/temporal holdout before production use.