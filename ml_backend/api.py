"""
REST API for Bundesliga Match Predictions
Provides endpoints for predictions, model info, and statistics
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from datetime import datetime
import traceback
from pathlib import Path

from data_collector import BundesligaDataCollector
from feature_engineering import FeatureEngineer
from models import BundesligaPredictor
from config import API_HOST, API_PORT, DATA_DIR

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables
predictor = None
feature_engineer = None
data_collector = None


def initialize_app():
    """Initialize models and data"""
    global predictor, feature_engineer, data_collector
    
    print("Initializing Bundesliga Predictor API...")
    
    # Initialize data collector
    data_collector = BundesligaDataCollector()
    
    # Load historical data
    matches_df = data_collector.load_data('sample_matches.csv')
    
    if matches_df.empty:
        print("No data found. Generating sample data...")
        from data_collector import generate_sample_data
        matches_df = generate_sample_data()
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(matches_df)
    
    # Initialize predictor
    predictor = BundesligaPredictor()
    
    # Try to load existing models
    try:
        predictor.load_models()
        print("Loaded existing models")
    except Exception as e:
        print(f"No existing models found. Training new models...")
        X, y = feature_engineer.create_training_dataset()
        predictor.train_all(X, y)
        predictor.save_models()
        print("New models trained and saved")
    
    print("API initialization complete!")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/predict', methods=['POST'])
def predict_match():
    """
    Predict match outcome
    
    Request body:
    {
        "home_team": "Bayern Munich",
        "away_team": "Borussia Dortmund",
        "match_date": "2024-03-15"  (optional, defaults to today)
    }
    
    Response:
    {
        "prediction": "HOME_WIN" | "DRAW" | "AWAY_WIN",
        "probabilities": {
            "home_win": 0.65,
            "draw": 0.20,
            "away_win": 0.15
        },
        "confidence": 0.65,
        "home_team": "Bayern Munich",
        "away_team": "Borussia Dortmund"
    }
    """
    try:
        data = request.get_json()
        
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        match_date_str = data.get('match_date', datetime.now().isoformat())
        
        if not home_team or not away_team:
            return jsonify({'error': 'home_team and away_team are required'}), 400
        
        # Parse date
        match_date = pd.to_datetime(match_date_str)
        
        # Create features
        features = feature_engineer.create_match_features(home_team, away_team, match_date)
        X = pd.DataFrame([features])
        
        # Make prediction
        prediction = predictor.predict(X, use_ensemble=True)[0]
        probabilities = predictor.predict_proba(X)[0]
        
        # Map prediction to label
        prediction_labels = {0: 'AWAY_WIN', 1: 'DRAW', 2: 'HOME_WIN'}
        prediction_label = prediction_labels[prediction]
        
        response = {
            'prediction': prediction_label,
            'probabilities': {
                'away_win': float(probabilities[0]),
                'draw': float(probabilities[1]),
                'home_win': float(probabilities[2])
            },
            'confidence': float(probabilities[prediction]),
            'home_team': home_team,
            'away_team': away_team,
            'match_date': match_date.isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in predict_match: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict multiple matches
    
    Request body:
    {
        "matches": [
            {"home_team": "Bayern Munich", "away_team": "Borussia Dortmund"},
            {"home_team": "RB Leipzig", "away_team": "Bayer Leverkusen"}
        ],
        "match_date": "2024-03-15"  (optional)
    }
    """
    try:
        data = request.get_json()
        matches = data.get('matches', [])
        match_date_str = data.get('match_date', datetime.now().isoformat())
        match_date = pd.to_datetime(match_date_str)
        
        predictions = []
        
        for match in matches:
            home_team = match.get('home_team')
            away_team = match.get('away_team')
            
            if not home_team or not away_team:
                continue
            
            features = feature_engineer.create_match_features(home_team, away_team, match_date)
            X = pd.DataFrame([features])
            
            prediction = predictor.predict(X, use_ensemble=True)[0]
            probabilities = predictor.predict_proba(X)[0]
            
            prediction_labels = {0: 'AWAY_WIN', 1: 'DRAW', 2: 'HOME_WIN'}
            
            predictions.append({
                'home_team': home_team,
                'away_team': away_team,
                'prediction': prediction_labels[prediction],
                'probabilities': {
                    'away_win': float(probabilities[0]),
                    'draw': float(probabilities[1]),
                    'home_win': float(probabilities[2])
                },
                'confidence': float(probabilities[prediction])
            })
        
        return jsonify({'predictions': predictions})
    
    except Exception as e:
        print(f"Error in predict_batch: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/teams', methods=['GET'])
def get_teams():
    """Get list of all teams"""
    try:
        teams = sorted(feature_engineer.teams)
        return jsonify({'teams': teams})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/team/stats', methods=['GET'])
def get_team_stats():
    """
    Get team statistics
    
    Query params:
    - team: Team name
    - before_date: Date to calculate stats before (optional)
    """
    try:
        team = request.args.get('team')
        before_date_str = request.args.get('before_date', datetime.now().isoformat())
        
        if not team:
            return jsonify({'error': 'team parameter is required'}), 400
        
        before_date = pd.to_datetime(before_date_str)
        
        # Get team statistics
        stats = feature_engineer.calculate_team_stats(team, before_date)
        form_5 = feature_engineer.calculate_form(team, before_date, 5)
        form_10 = feature_engineer.calculate_form(team, before_date, 10)
        
        return jsonify({
            'team': team,
            'stats': stats,
            'recent_form_5': form_5,
            'recent_form_10': form_10
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/head-to-head', methods=['GET'])
def get_head_to_head():
    """
    Get head-to-head statistics between two teams
    
    Query params:
    - team1: First team name
    - team2: Second team name
    - before_date: Date to calculate stats before (optional)
    """
    try:
        team1 = request.args.get('team1')
        team2 = request.args.get('team2')
        before_date_str = request.args.get('before_date', datetime.now().isoformat())
        
        if not team1 or not team2:
            return jsonify({'error': 'team1 and team2 parameters are required'}), 400
        
        before_date = pd.to_datetime(before_date_str)
        
        h2h = feature_engineer.calculate_head_to_head(team1, team2, before_date)
        
        return jsonify({
            'team1': team1,
            'team2': team2,
            'head_to_head': h2h
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Get information about the models"""
    try:
        feature_importance = predictor.get_feature_importance(top_n=15)
        
        return jsonify({
            'models': ['Random Forest', 'XGBoost', 'Neural Network', 'Ensemble'],
            'feature_count': len(predictor.feature_names),
            'top_features': feature_importance.to_dict('records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/standings', methods=['GET'])
def get_standings():
    """Get current Bundesliga standings"""
    try:
        standings = data_collector.fetch_standings()
        
        if not standings.empty:
            return jsonify({'standings': standings.to_dict('records')})
        else:
            return jsonify({'error': 'Could not fetch standings'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upcoming', methods=['GET'])
def get_upcoming_matches():
    """Get upcoming Bundesliga matches with predictions"""
    try:
        upcoming = data_collector.fetch_matches(status='SCHEDULED')
        
        if upcoming.empty:
            return jsonify({'matches': []})
        
        # Add predictions for upcoming matches
        predictions = []
        for _, match in upcoming.head(10).iterrows():  # Limit to 10 matches
            home_team = match['home_team']
            away_team = match['away_team']
            match_date = pd.to_datetime(match['date'])
            
            features = feature_engineer.create_match_features(home_team, away_team, match_date)
            X = pd.DataFrame([features])
            
            prediction = predictor.predict(X, use_ensemble=True)[0]
            probabilities = predictor.predict_proba(X)[0]
            
            prediction_labels = {0: 'AWAY_WIN', 1: 'DRAW', 2: 'HOME_WIN'}
            
            predictions.append({
                'match_id': match['match_id'],
                'date': match['date'],
                'home_team': home_team,
                'away_team': away_team,
                'prediction': prediction_labels[prediction],
                'probabilities': {
                    'away_win': float(probabilities[0]),
                    'draw': float(probabilities[1]),
                    'home_win': float(probabilities[2])
                }
            })
        
        return jsonify({'matches': predictions})
    
    except Exception as e:
        print(f"Error in get_upcoming_matches: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    initialize_app()
    app.run(host=API_HOST, port=API_PORT, debug=True)
