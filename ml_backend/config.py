"""
Configuration for Bundesliga ML Predictor
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# API Configuration
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 5000))

# Football Data API (you'll need to get a free API key from https://www.football-data.org/)
FOOTBALL_DATA_API_KEY = os.getenv('FOOTBALL_DATA_API_KEY', 'YOUR_API_KEY_HERE')
FOOTBALL_DATA_BASE_URL = 'https://api.football-data.org/v4'

# Bundesliga specific
BUNDESLIGA_ID = 2002  # Bundesliga league ID in football-data.org API

# Model Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.2

# Features to use for prediction
FEATURE_CATEGORIES = {
    'team_stats': [
        'goals_scored', 'goals_conceded', 'goal_difference',
        'wins', 'draws', 'losses', 'points',
        'home_goals_scored', 'home_goals_conceded',
        'away_goals_scored', 'away_goals_conceded'
    ],
    'form': [
        'last_5_wins', 'last_5_goals_scored', 'last_5_goals_conceded',
        'last_10_wins', 'last_10_goals_scored', 'last_10_goals_conceded',
        'win_streak', 'unbeaten_streak'
    ],
    'head_to_head': [
        'h2h_wins', 'h2h_draws', 'h2h_losses',
        'h2h_goals_scored', 'h2h_goals_conceded',
        'h2h_last_5_wins'
    ],
    'advanced': [
        'average_possession', 'shots_per_game', 'shots_on_target_per_game',
        'corners_per_game', 'fouls_per_game', 'yellow_cards_per_game',
        'red_cards_per_game', 'offsides_per_game'
    ]
}

# Model hyperparameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'random_state': RANDOM_STATE
    },
    'xgboost': {
        'n_estimators': 150,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE
    },
    'neural_network': {
        'hidden_layers': [128, 64, 32],
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32
    }
}
