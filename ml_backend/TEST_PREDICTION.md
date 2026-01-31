# 🧪 Test Your Predictions

Quick commands to test the Bundesliga prediction system.

## 1️⃣ Test via Python Script

```bash
cd ml_backend
python example_usage.py
```

This runs 5 comprehensive examples showing all system capabilities.

## 2️⃣ Test via API (cURL)

### Start the API
```bash
python api.py
```

### Health Check
```bash
curl http://localhost:5000/health
```

### Predict a Match
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Bayern Munich",
    "away_team": "Borussia Dortmund"
  }'
```

### Get All Teams
```bash
curl http://localhost:5000/api/teams
```

### Get Team Statistics
```bash
curl "http://localhost:5000/api/team/stats?team=Bayern%20Munich"
```

### Get Upcoming Matches
```bash
curl http://localhost:5000/api/upcoming
```

## 3️⃣ Test via Python (Interactive)

```python
from data_collector import generate_sample_data
from feature_engineering import FeatureEngineer
from models import BundesligaPredictor
import pandas as pd
from datetime import datetime

# Setup
matches_df = generate_sample_data()
fe = FeatureEngineer(matches_df)
predictor = BundesligaPredictor()

# Train or load
try:
    predictor.load_models()
except:
    X, y = fe.create_training_dataset()
    predictor.train_all(X, y)
    predictor.save_models()

# Predict
features = fe.create_match_features(
    "Bayern Munich", 
    "Borussia Dortmund", 
    datetime.now()
)
X = pd.DataFrame([features])

prediction = predictor.predict(X)[0]
probabilities = predictor.predict_proba(X)[0]

result = ['Away Win', 'Draw', 'Home Win'][prediction]
print(f"Prediction: {result}")
print(f"Confidence: {probabilities[prediction]:.1%}")
```

## 4️⃣ Test via Web UI

1. Start backend: `python api.py`
2. Start frontend: `npm run dev` (in project root)
3. Open: `http://localhost:5173/bundesliga`
4. Select teams and click "Predict Match"

## 5️⃣ Evaluate Models

```bash
python evaluate.py
```

Results saved to: `data/evaluation/`
- Confusion matrices
- Feature importance
- Model comparison charts
- Detailed report

## ✅ Expected Results

**Sample prediction for Bayern Munich vs Borussia Dortmund:**
- Home Win: ~60-70%
- Draw: ~15-25%
- Away Win: ~10-20%

**Model Accuracies (with sample data):**
- Random Forest: ~55-65%
- XGBoost: ~60-70%
- Neural Network: ~55-65%
- Ensemble: ~65-75%

*Actual values depend on the generated sample data*

## 🐛 Common Issues

**"No module named 'sklearn'"**
```bash
pip install -r requirements.txt
```

**"Models not found"**
```bash
python train.py
```

**"Connection refused"**
- Ensure API is running: `python api.py`
- Check port 5000 is not in use

**"Team not found"**
- Check team name spelling
- Get valid teams: `curl http://localhost:5000/api/teams`

## 📊 Batch Testing

Test multiple matches at once:

```bash
curl -X POST http://localhost:5000/api/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "matches": [
      {"home_team": "Bayern Munich", "away_team": "RB Leipzig"},
      {"home_team": "Borussia Dortmund", "away_team": "Bayer Leverkusen"},
      {"home_team": "Union Berlin", "away_team": "SC Freiburg"}
    ]
  }'
```

## 🎯 Verify Installation

Run all verification steps:

```bash
# 1. Check Python dependencies
pip list | grep -E "(sklearn|xgboost|tensorflow|flask|pandas)"

# 2. Verify data exists
ls -la data/

# 3. Check models trained
ls -la models/

# 4. Test import
python -c "from models import BundesligaPredictor; print('✓ Import successful')"

# 5. Run health check
python -c "from data_collector import generate_sample_data; df = generate_sample_data(); print(f'✓ Generated {len(df)} matches')"
```

## 🚀 Next Steps

- Try different team combinations
- Adjust model hyperparameters in `config.py`
- Add new features in `feature_engineering.py`
- Use real data with Football-Data.org API key

---

Happy Predicting! ⚽🤖
