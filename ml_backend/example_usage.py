"""
Example usage of the Bundesliga prediction system
Run this script to see the system in action
"""
import pandas as pd
from datetime import datetime

from data_collector import BundesligaDataCollector, generate_sample_data
from feature_engineering import FeatureEngineer
from models import BundesligaPredictor


def example_single_prediction():
    """Example: Predict a single match"""
    print("=" * 70)
    print("EXAMPLE 1: Single Match Prediction")
    print("=" * 70)
    
    # Load data and models
    matches_df = BundesligaDataCollector().load_data('sample_matches.csv')
    if matches_df.empty:
        matches_df = generate_sample_data()
    
    fe = FeatureEngineer(matches_df)
    predictor = BundesligaPredictor()
    
    try:
        predictor.load_models()
    except:
        print("Training models first...")
        X, y = fe.create_training_dataset()
        predictor.train_all(X, y)
        predictor.save_models()
    
    # Make prediction
    home_team = "Bayern Munich"
    away_team = "Borussia Dortmund"
    match_date = datetime.now()
    
    print(f"\nPredicting: {home_team} vs {away_team}")
    print(f"Date: {match_date.strftime('%Y-%m-%d')}")
    
    features = fe.create_match_features(home_team, away_team, match_date)
    X = pd.DataFrame([features])
    
    prediction = predictor.predict(X, use_ensemble=True)[0]
    probabilities = predictor.predict_proba(X)[0]
    
    result_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
    
    print(f"\n🎯 PREDICTION: {result_map[prediction]}")
    print(f"\n📊 PROBABILITIES:")
    print(f"   Home Win: {probabilities[2]:.1%}")
    print(f"   Draw:     {probabilities[1]:.1%}")
    print(f"   Away Win: {probabilities[0]:.1%}")
    print(f"\n✅ Confidence: {probabilities[prediction]:.1%}")


def example_batch_predictions():
    """Example: Predict multiple matches"""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: Batch Predictions")
    print("=" * 70)
    
    # Load data and models
    matches_df = BundesligaDataCollector().load_data('sample_matches.csv')
    if matches_df.empty:
        matches_df = generate_sample_data()
    
    fe = FeatureEngineer(matches_df)
    predictor = BundesligaPredictor()
    predictor.load_models()
    
    # Matches to predict
    upcoming_matches = [
        ("Bayern Munich", "RB Leipzig"),
        ("Borussia Dortmund", "Bayer Leverkusen"),
        ("Union Berlin", "SC Freiburg"),
        ("VfL Wolfsburg", "Eintracht Frankfurt")
    ]
    
    match_date = datetime.now()
    
    print("\n📅 Upcoming Match Predictions:\n")
    
    for home, away in upcoming_matches:
        features = fe.create_match_features(home, away, match_date)
        X = pd.DataFrame([features])
        
        prediction = predictor.predict(X)[0]
        probabilities = predictor.predict_proba(X)[0]
        
        result_map = {0: '🔴 Away Win', 1: '🟡 Draw', 2: '🟢 Home Win'}
        
        print(f"{home:25s} vs {away:25s}")
        print(f"  → {result_map[prediction]} ({probabilities[prediction]:.1%} confidence)")
        print()


def example_team_statistics():
    """Example: Get team statistics"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Team Statistics")
    print("=" * 70)
    
    matches_df = BundesligaDataCollector().load_data('sample_matches.csv')
    if matches_df.empty:
        matches_df = generate_sample_data()
    
    fe = FeatureEngineer(matches_df)
    
    team = "Bayern Munich"
    date = datetime.now()
    
    print(f"\n📊 Statistics for {team}:\n")
    
    # Overall stats
    stats = fe.calculate_team_stats(team, date)
    print("OVERALL PERFORMANCE:")
    print(f"  Goals Scored:    {stats['goals_scored']}")
    print(f"  Goals Conceded:  {stats['goals_conceded']}")
    print(f"  Goal Difference: {stats['goal_difference']:+d}")
    print(f"  Wins:            {stats['wins']}")
    print(f"  Draws:           {stats['draws']}")
    print(f"  Losses:          {stats['losses']}")
    print(f"  Points:          {stats['points']}")
    print(f"  Win Rate:        {stats['win_rate']:.1%}")
    
    # Recent form
    form = fe.calculate_form(team, date, 5)
    print(f"\nRECENT FORM (Last 5 matches):")
    print(f"  Wins:            {form['last_5_wins']}")
    print(f"  Goals Scored:    {form['last_5_goals_scored']}")
    print(f"  Goals Conceded:  {form['last_5_goals_conceded']}")
    print(f"  Points:          {form['last_5_points']}")
    print(f"  Win Streak:      {form['win_streak']}")
    print(f"  Unbeaten Streak: {form['unbeaten_streak']}")


def example_head_to_head():
    """Example: Head-to-head statistics"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Head-to-Head Statistics")
    print("=" * 70)
    
    matches_df = BundesligaDataCollector().load_data('sample_matches.csv')
    if matches_df.empty:
        matches_df = generate_sample_data()
    
    fe = FeatureEngineer(matches_df)
    
    team1 = "Bayern Munich"
    team2 = "Borussia Dortmund"
    date = datetime.now()
    
    print(f"\n⚔️  {team1} vs {team2}\n")
    
    h2h = fe.calculate_head_to_head(team1, team2, date)
    
    print(f"Total Matches:        {h2h['h2h_matches']}")
    print(f"{team1} Wins:  {h2h['h2h_team1_wins']}")
    print(f"Draws:                {h2h['h2h_draws']}")
    print(f"{team2} Wins:  {h2h['h2h_team2_wins']}")
    print(f"\nGoals:")
    print(f"  {team1}: {h2h['h2h_team1_goals']}")
    print(f"  {team2}: {h2h['h2h_team2_goals']}")
    print(f"\n{team1} Win Rate: {h2h['h2h_team1_win_rate']:.1%}")


def example_feature_importance():
    """Example: Show most important features"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Feature Importance")
    print("=" * 70)
    
    matches_df = BundesligaDataCollector().load_data('sample_matches.csv')
    if matches_df.empty:
        matches_df = generate_sample_data()
    
    fe = FeatureEngineer(matches_df)
    predictor = BundesligaPredictor()
    
    try:
        predictor.load_models()
    except:
        print("Training models first...")
        X, y = fe.create_training_dataset()
        predictor.train_all(X, y)
        predictor.save_models()
    
    print("\n🔍 Top 10 Most Important Features for Predictions:\n")
    
    importance_df = predictor.get_feature_importance(top_n=10)
    
    for idx, row in importance_df.iterrows():
        bar_length = int(row['importance'] * 50)
        bar = '█' * bar_length
        print(f"{idx+1:2d}. {row['feature']:35s} {bar} {row['importance']:.4f}")


def main():
    """Run all examples"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║        BUNDESLIGA MATCH PREDICTION - EXAMPLE USAGE               ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    example_single_prediction()
    example_batch_predictions()
    example_team_statistics()
    example_head_to_head()
    example_feature_importance()
    
    print("\n" + "=" * 70)
    print("All examples completed successfully! 🎉")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Start the API: python api.py")
    print("  2. Start React frontend: npm run dev")
    print("  3. Visit http://localhost:5173/bundesliga")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
