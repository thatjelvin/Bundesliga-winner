"""
Predict the winner of the 2025-26 Bundesliga season
Simulates remaining matches and calculates final standings
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

from data_collector import BundesligaDataCollector, generate_sample_data
from feature_engineering import FeatureEngineer
from models import BundesligaPredictor


class SeasonWinnerPredictor:
    """Predict season winner by simulating remaining matches"""
    
    def __init__(self, predictor: BundesligaPredictor, feature_engineer: FeatureEngineer):
        self.predictor = predictor
        self.fe = feature_engineer
        self.teams = sorted(self.fe.teams)
    
    def calculate_current_standings(self, season: str = "2025") -> pd.DataFrame:
        """Calculate current standings based on completed matches"""
        standings = defaultdict(lambda: {
            'played': 0, 'won': 0, 'drawn': 0, 'lost': 0,
            'goals_for': 0, 'goals_against': 0, 'points': 0
        })
        
        # Get completed matches for the current season
        current_season_matches = self.fe.matches_df[
            (self.fe.matches_df['season'] == int(season)) &
            (self.fe.matches_df['status'] == 'FINISHED')
        ]
        
        for _, match in current_season_matches.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            home_score = match['home_score']
            away_score = match['away_score']
            
            # Update home team stats
            standings[home_team]['played'] += 1
            standings[home_team]['goals_for'] += home_score
            standings[home_team]['goals_against'] += away_score
            
            # Update away team stats
            standings[away_team]['played'] += 1
            standings[away_team]['goals_for'] += away_score
            standings[away_team]['goals_against'] += home_score
            
            # Update win/draw/loss
            if home_score > away_score:
                standings[home_team]['won'] += 1
                standings[home_team]['points'] += 3
                standings[away_team]['lost'] += 1
            elif away_score > home_score:
                standings[away_team]['won'] += 1
                standings[away_team]['points'] += 3
                standings[home_team]['lost'] += 1
            else:
                standings[home_team]['drawn'] += 1
                standings[home_team]['points'] += 1
                standings[away_team]['drawn'] += 1
                standings[away_team]['points'] += 1
        
        # Convert to DataFrame
        standings_list = []
        for team, stats in standings.items():
            stats['team'] = team
            stats['goal_difference'] = stats['goals_for'] - stats['goals_against']
            standings_list.append(stats)
        
        df = pd.DataFrame(standings_list)
        if df.empty:
            # Create default standings for all teams
            df = pd.DataFrame([
                {
                    'team': team,
                    'played': 0, 'won': 0, 'drawn': 0, 'lost': 0,
                    'goals_for': 0, 'goals_against': 0, 'goal_difference': 0, 'points': 0
                }
                for team in self.teams
            ])
        
        # Sort by points, then goal difference, then goals scored
        df = df.sort_values(
            by=['points', 'goal_difference', 'goals_for'],
            ascending=[False, False, False]
        ).reset_index(drop=True)
        
        return df
    
    def generate_remaining_fixtures(self, season: str = "2025") -> List[Tuple[str, str]]:
        """Generate all remaining fixtures for the season"""
        # In a full Bundesliga season, each team plays every other team twice (home and away)
        # That's 18 teams * 17 opponents * 2 = 612 total matches in a season
        # Or 18 * 17 = 306 matches (each match counted once)
        
        all_fixtures = []
        
        # Generate all possible home/away combinations
        for home_team in self.teams:
            for away_team in self.teams:
                if home_team != away_team:
                    all_fixtures.append((home_team, away_team))
        
        # Get already played matches
        played_matches = self.fe.matches_df[
            (self.fe.matches_df['season'] == int(season)) &
            (self.fe.matches_df['status'] == 'FINISHED')
        ]
        
        played_pairs = set()
        for _, match in played_matches.iterrows():
            played_pairs.add((match['home_team'], match['away_team']))
        
        # Filter out played matches
        remaining_fixtures = [
            fixture for fixture in all_fixtures
            if fixture not in played_pairs
        ]
        
        return remaining_fixtures
    
    def simulate_match(self, home_team: str, away_team: str, 
                      match_date: datetime = None) -> Tuple[int, int, str]:
        """
        Simulate a single match and return predicted score
        
        Returns:
            Tuple of (home_score, away_score, result)
        """
        if match_date is None:
            match_date = datetime.now()
        
        # Create features for the match
        features = self.fe.create_match_features(home_team, away_team, match_date)
        X = pd.DataFrame([features])
        
        # Get prediction - use only XGBoost for speed
        X_scaled = self.predictor.scaler.transform(X)
        prediction = self.predictor.models['xgboost'].predict(X_scaled)[0]
        probabilities = self.predictor.models['xgboost'].predict_proba(X_scaled)[0]
        
        # Map prediction to result
        result_map = {0: 'AWAY_WIN', 1: 'DRAW', 2: 'HOME_WIN'}
        result = result_map[prediction]
        
        # Generate realistic scores based on prediction
        # Use team statistics to estimate scores
        home_stats = self.fe.calculate_team_stats(home_team, match_date)
        away_stats = self.fe.calculate_team_stats(away_team, match_date)
        
        home_avg_goals = max(0.5, home_stats['avg_goals_scored'])
        away_avg_goals = max(0.5, away_stats['avg_goals_scored'])
        
        if result == 'HOME_WIN':
            home_score = max(1, int(np.random.poisson(home_avg_goals + 0.5)))
            away_score = max(0, int(np.random.poisson(away_avg_goals - 0.3)))
            # Ensure home wins
            if away_score >= home_score:
                home_score = away_score + 1
        elif result == 'AWAY_WIN':
            home_score = max(0, int(np.random.poisson(home_avg_goals - 0.3)))
            away_score = max(1, int(np.random.poisson(away_avg_goals + 0.5)))
            # Ensure away wins
            if home_score >= away_score:
                away_score = home_score + 1
        else:  # DRAW
            score = max(0, int(np.random.poisson((home_avg_goals + away_avg_goals) / 2)))
            home_score = away_score = score
        
        return home_score, away_score, result
    
    def _print_standings(self, standings: pd.DataFrame, top_n: int = None):
        """Print standings table in a nice format"""
        if top_n:
            standings = standings.head(top_n)
        
        print(f"{'Pos':<4} {'Team':<30} {'P':<4} {'W':<4} {'D':<4} {'L':<4} {'GF':<4} {'GA':<4} {'GD':<5} {'Pts':<4}")
        print("-" * 70)
        
        for idx, row in standings.iterrows():
            pos = idx + 1
            print(f"{pos:<4} {row['team']:<30} {int(row['played']):<4} {int(row['won']):<4} "
                  f"{int(row['drawn']):<4} {int(row['lost']):<4} {int(row['goals_for']):<4} "
                  f"{int(row['goals_against']):<4} {int(row['goal_difference']):<5} {int(row['points']):<4}")
    
    def simulate_season(self, n_simulations: int = 1000, season: str = "2025") -> Dict:
        """
        Simulate the rest of the season multiple times
        
        Args:
            n_simulations: Number of times to simulate the season
            season: Season year (e.g., "2025" for 2025-26)
            
        Returns:
            Dictionary with simulation results and predictions
        """
        print(f"\n{'=' * 70}")
        print(f"SIMULATING 2025-26 BUNDESLIGA SEASON")
        print(f"{'=' * 70}\n")
        
        # Get current standings
        current_standings = self.calculate_current_standings(season)
        print("Current Standings:")
        print("-" * 70)
        self._print_standings(current_standings.head(10))
        
        # Get remaining fixtures
        remaining_fixtures = self.generate_remaining_fixtures(season)
        print(f"\nRemaining fixtures: {len(remaining_fixtures)}")
        
        if len(remaining_fixtures) == 0:
            print("\nNo remaining fixtures. Season is complete!")
            print(f"\n🏆 BUNDESLIGA 2025-26 CHAMPION: {current_standings.iloc[0]['team']}")
            return {
                'champion': current_standings.iloc[0]['team'],
                'final_standings': current_standings,
                'championship_probability': {current_standings.iloc[0]['team']: 1.0}
            }
        
        # Run simulations
        print(f"\nRunning {n_simulations} simulations...")
        championship_wins = defaultdict(int)
        top_4_finishes = defaultdict(int)
        
        for sim in range(n_simulations):
            # Start with current standings
            sim_standings = current_standings.copy()
            
            # Simulate remaining matches
            for home_team, away_team in remaining_fixtures:
                home_score, away_score, result = self.simulate_match(
                    home_team, away_team, datetime.now()
                )
                
                # Update standings
                home_idx = sim_standings[sim_standings['team'] == home_team].index[0]
                away_idx = sim_standings[sim_standings['team'] == away_team].index[0]
                
                sim_standings.loc[home_idx, 'played'] += 1
                sim_standings.loc[away_idx, 'played'] += 1
                sim_standings.loc[home_idx, 'goals_for'] += home_score
                sim_standings.loc[home_idx, 'goals_against'] += away_score
                sim_standings.loc[away_idx, 'goals_for'] += away_score
                sim_standings.loc[away_idx, 'goals_against'] += home_score
                
                if home_score > away_score:
                    sim_standings.loc[home_idx, 'won'] += 1
                    sim_standings.loc[home_idx, 'points'] += 3
                    sim_standings.loc[away_idx, 'lost'] += 1
                elif away_score > home_score:
                    sim_standings.loc[away_idx, 'won'] += 1
                    sim_standings.loc[away_idx, 'points'] += 3
                    sim_standings.loc[home_idx, 'lost'] += 1
                else:
                    sim_standings.loc[home_idx, 'drawn'] += 1
                    sim_standings.loc[home_idx, 'points'] += 1
                    sim_standings.loc[away_idx, 'drawn'] += 1
                    sim_standings.loc[away_idx, 'points'] += 1
                
                # Update goal difference
                sim_standings.loc[home_idx, 'goal_difference'] = \
                    sim_standings.loc[home_idx, 'goals_for'] - sim_standings.loc[home_idx, 'goals_against']
                sim_standings.loc[away_idx, 'goal_difference'] = \
                    sim_standings.loc[away_idx, 'goals_for'] - sim_standings.loc[away_idx, 'goals_against']
            
            # Sort final standings
            sim_standings = sim_standings.sort_values(
                by=['points', 'goal_difference', 'goals_for'],
                ascending=[False, False, False]
            ).reset_index(drop=True)
            
            # Record champion
            champion = sim_standings.iloc[0]['team']
            championship_wins[champion] += 1
            
            # Record top 4
            for i in range(min(4, len(sim_standings))):
                top_4_finishes[sim_standings.iloc[i]['team']] += 1
        
        # Calculate probabilities
        championship_prob = {
            team: (wins / n_simulations) * 100
            for team, wins in championship_wins.items()
        }
        
        top_4_prob = {
            team: (finishes / n_simulations) * 100
            for team, finishes in top_4_finishes.items()
        }
        
        # Sort by probability
        sorted_championship = sorted(
            championship_prob.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Print results
        print(f"\n{'=' * 70}")
        print("CHAMPIONSHIP PREDICTIONS (2025-26 Season)")
        print(f"{'=' * 70}\n")
        
        print("🏆 Title Race Probabilities:\n")
        for i, (team, prob) in enumerate(sorted_championship[:10], 1):
            bar_length = int(prob / 2)
            bar = '█' * bar_length
            print(f"{i:2d}. {team:30s} {bar} {prob:5.2f}%")
        
        predicted_champion = sorted_championship[0][0]
        champion_prob = sorted_championship[0][1]
        
        print(f"\n{'=' * 70}")
        print(f"🏆 PREDICTED CHAMPION: {predicted_champion}")
        print(f"   Probability: {champion_prob:.2f}%")
        print(f"{'=' * 70}\n")
        
        return {
            'champion': predicted_champion,
            'championship_probability': dict(sorted_championship),
            'top_4_probability': top_4_prob,
            'current_standings': current_standings,
            'n_simulations': n_simulations,
            'remaining_fixtures': len(remaining_fixtures)
        }


def main():
    """Main function to predict season winner"""
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "BUNDESLIGA 2025-26 SEASON WINNER PREDICTION" + " " * 10 + "║")
    print("╚" + "=" * 68 + "╝\n")
    
    # Load data
    print("Loading data and models...")
    collector = BundesligaDataCollector()
    matches_df = collector.load_data('sample_matches.csv')
    
    if matches_df.empty:
        print("Generating sample data...")
        matches_df = generate_sample_data()
    
    # Initialize feature engineer
    fe = FeatureEngineer(matches_df)
    
    # Load predictor
    predictor = BundesligaPredictor()
    
    try:
        predictor.load_models()
        print("✓ Models loaded successfully\n")
    except:
        print("Training models...")
        X, y = fe.create_training_dataset()
        predictor.train_all(X, y)
        predictor.save_models()
        print("✓ Models trained and saved\n")
    
    # Create season predictor
    season_predictor = SeasonWinnerPredictor(predictor, fe)
    
    # Simulate season
    results = season_predictor.simulate_season(n_simulations=10, season="2025")
    
    print("\nPrediction complete! 🎉")
    print("\nNote: Predictions are based on historical data and current form.")
    print("Actual results may vary due to injuries, transfers, and other factors.")


if __name__ == '__main__':
    main()
