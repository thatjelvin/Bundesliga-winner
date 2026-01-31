"""
Feature Engineering for Bundesliga Match Prediction
Extracts and engineers features from historical match data
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta


class FeatureEngineer:
    """Engineers features for machine learning from raw match data"""
    
    def __init__(self, matches_df: pd.DataFrame):
        """
        Initialize with historical match data
        
        Args:
            matches_df: DataFrame with columns [match_id, date, home_team, away_team, 
                       home_score, away_score, winner, season, matchday]
        """
        self.matches_df = matches_df.copy()
        self.matches_df['date'] = pd.to_datetime(self.matches_df['date'])
        self.matches_df = self.matches_df.sort_values('date')
        
        # Team mapping
        self.teams = sorted(set(self.matches_df['home_team'].unique()) | 
                           set(self.matches_df['away_team'].unique()))
        self.team_to_id = {team: idx for idx, team in enumerate(self.teams)}
    
    def calculate_team_stats(self, team: str, before_date: datetime, 
                            window_days: int = 365) -> Dict[str, float]:
        """
        Calculate team statistics before a specific date
        
        Args:
            team: Team name
            before_date: Calculate stats before this date
            window_days: Number of days to look back
            
        Returns:
            Dictionary of team statistics
        """
        cutoff_date = before_date - timedelta(days=window_days)
        
        # Filter matches for this team within the window
        team_matches = self.matches_df[
            (self.matches_df['date'] < before_date) &
            (self.matches_df['date'] >= cutoff_date) &
            ((self.matches_df['home_team'] == team) | 
             (self.matches_df['away_team'] == team))
        ].copy()
        
        if len(team_matches) == 0:
            return self._get_default_stats()
        
        # Calculate statistics
        stats = {}
        
        # Home matches
        home_matches = team_matches[team_matches['home_team'] == team]
        # Away matches
        away_matches = team_matches[team_matches['away_team'] == team]
        
        # Goals scored and conceded
        home_goals_scored = home_matches['home_score'].sum()
        home_goals_conceded = home_matches['away_score'].sum()
        away_goals_scored = away_matches['away_score'].sum()
        away_goals_conceded = away_matches['home_score'].sum()
        
        stats['goals_scored'] = home_goals_scored + away_goals_scored
        stats['goals_conceded'] = home_goals_conceded + away_goals_conceded
        stats['goal_difference'] = stats['goals_scored'] - stats['goals_conceded']
        
        # Home/Away specific
        stats['home_goals_scored'] = home_goals_scored
        stats['home_goals_conceded'] = home_goals_conceded
        stats['away_goals_scored'] = away_goals_scored
        stats['away_goals_conceded'] = away_goals_conceded
        
        # Wins, draws, losses
        home_wins = len(home_matches[home_matches['winner'] == 'HOME_TEAM'])
        home_draws = len(home_matches[home_matches['winner'] == 'DRAW'])
        away_wins = len(away_matches[away_matches['winner'] == 'AWAY_TEAM'])
        away_draws = len(away_matches[away_matches['winner'] == 'DRAW'])
        
        stats['wins'] = home_wins + away_wins
        stats['draws'] = home_draws + away_draws
        stats['losses'] = len(team_matches) - stats['wins'] - stats['draws']
        stats['points'] = stats['wins'] * 3 + stats['draws']
        
        # Win/draw/loss ratios
        total_matches = len(team_matches)
        stats['win_rate'] = stats['wins'] / total_matches if total_matches > 0 else 0
        stats['draw_rate'] = stats['draws'] / total_matches if total_matches > 0 else 0
        stats['loss_rate'] = stats['losses'] / total_matches if total_matches > 0 else 0
        
        # Averages per game
        stats['avg_goals_scored'] = stats['goals_scored'] / total_matches if total_matches > 0 else 0
        stats['avg_goals_conceded'] = stats['goals_conceded'] / total_matches if total_matches > 0 else 0
        stats['avg_points'] = stats['points'] / total_matches if total_matches > 0 else 0
        
        return stats
    
    def calculate_form(self, team: str, before_date: datetime, 
                      last_n_matches: int = 5) -> Dict[str, float]:
        """
        Calculate recent form statistics
        
        Args:
            team: Team name
            before_date: Calculate form before this date
            last_n_matches: Number of recent matches to consider
            
        Returns:
            Dictionary of form statistics
        """
        # Get last N matches before the date
        team_matches = self.matches_df[
            (self.matches_df['date'] < before_date) &
            ((self.matches_df['home_team'] == team) | 
             (self.matches_df['away_team'] == team))
        ].tail(last_n_matches).copy()
        
        if len(team_matches) == 0:
            return {
                f'last_{last_n_matches}_wins': 0,
                f'last_{last_n_matches}_draws': 0,
                f'last_{last_n_matches}_losses': 0,
                f'last_{last_n_matches}_goals_scored': 0,
                f'last_{last_n_matches}_goals_conceded': 0,
                f'last_{last_n_matches}_points': 0,
                'win_streak': 0,
                'unbeaten_streak': 0,
                'losing_streak': 0
            }
        
        form = {}
        
        # Determine results for each match
        results = []
        goals_scored = []
        goals_conceded = []
        
        for _, match in team_matches.iterrows():
            if match['home_team'] == team:
                goals_scored.append(match['home_score'])
                goals_conceded.append(match['away_score'])
                if match['winner'] == 'HOME_TEAM':
                    results.append('W')
                elif match['winner'] == 'DRAW':
                    results.append('D')
                else:
                    results.append('L')
            else:  # away team
                goals_scored.append(match['away_score'])
                goals_conceded.append(match['home_score'])
                if match['winner'] == 'AWAY_TEAM':
                    results.append('W')
                elif match['winner'] == 'DRAW':
                    results.append('D')
                else:
                    results.append('L')
        
        # Calculate form statistics
        form[f'last_{last_n_matches}_wins'] = results.count('W')
        form[f'last_{last_n_matches}_draws'] = results.count('D')
        form[f'last_{last_n_matches}_losses'] = results.count('L')
        form[f'last_{last_n_matches}_goals_scored'] = sum(goals_scored)
        form[f'last_{last_n_matches}_goals_conceded'] = sum(goals_conceded)
        form[f'last_{last_n_matches}_points'] = results.count('W') * 3 + results.count('D')
        
        # Calculate streaks
        form['win_streak'] = self._calculate_streak(results, 'W')
        form['unbeaten_streak'] = self._calculate_unbeaten_streak(results)
        form['losing_streak'] = self._calculate_streak(results, 'L')
        
        return form
    
    def calculate_head_to_head(self, team1: str, team2: str, 
                              before_date: datetime, last_n: int = 10) -> Dict[str, float]:
        """
        Calculate head-to-head statistics between two teams
        
        Args:
            team1: First team (typically home team)
            team2: Second team (typically away team)
            before_date: Calculate stats before this date
            last_n: Number of recent H2H matches to consider
            
        Returns:
            Dictionary of H2H statistics
        """
        # Get head-to-head matches
        h2h_matches = self.matches_df[
            (self.matches_df['date'] < before_date) &
            (((self.matches_df['home_team'] == team1) & (self.matches_df['away_team'] == team2)) |
             ((self.matches_df['home_team'] == team2) & (self.matches_df['away_team'] == team1)))
        ].tail(last_n)
        
        if len(h2h_matches) == 0:
            return {
                'h2h_matches': 0,
                'h2h_team1_wins': 0,
                'h2h_draws': 0,
                'h2h_team2_wins': 0,
                'h2h_team1_goals': 0,
                'h2h_team2_goals': 0,
                'h2h_team1_win_rate': 0
            }
        
        h2h = {'h2h_matches': len(h2h_matches)}
        
        team1_wins = 0
        team2_wins = 0
        draws = 0
        team1_goals = 0
        team2_goals = 0
        
        for _, match in h2h_matches.iterrows():
            if match['home_team'] == team1:
                team1_goals += match['home_score']
                team2_goals += match['away_score']
                if match['winner'] == 'HOME_TEAM':
                    team1_wins += 1
                elif match['winner'] == 'AWAY_TEAM':
                    team2_wins += 1
                else:
                    draws += 1
            else:
                team1_goals += match['away_score']
                team2_goals += match['home_score']
                if match['winner'] == 'AWAY_TEAM':
                    team1_wins += 1
                elif match['winner'] == 'HOME_TEAM':
                    team2_wins += 1
                else:
                    draws += 1
        
        h2h['h2h_team1_wins'] = team1_wins
        h2h['h2h_draws'] = draws
        h2h['h2h_team2_wins'] = team2_wins
        h2h['h2h_team1_goals'] = team1_goals
        h2h['h2h_team2_goals'] = team2_goals
        h2h['h2h_team1_win_rate'] = team1_wins / len(h2h_matches) if len(h2h_matches) > 0 else 0
        
        return h2h
    
    def create_match_features(self, home_team: str, away_team: str, 
                             match_date: datetime) -> Dict[str, float]:
        """
        Create all features for a single match prediction
        
        Args:
            home_team: Home team name
            away_team: Away team name
            match_date: Date of the match
            
        Returns:
            Dictionary of all features
        """
        features = {}
        
        # Team statistics (last 365 days)
        home_stats = self.calculate_team_stats(home_team, match_date)
        away_stats = self.calculate_team_stats(away_team, match_date)
        
        # Add with prefixes
        for key, value in home_stats.items():
            features[f'home_{key}'] = value
        for key, value in away_stats.items():
            features[f'away_{key}'] = value
        
        # Recent form (last 5 matches)
        home_form_5 = self.calculate_form(home_team, match_date, 5)
        away_form_5 = self.calculate_form(away_team, match_date, 5)
        
        for key, value in home_form_5.items():
            features[f'home_{key}'] = value
        for key, value in away_form_5.items():
            features[f'away_{key}'] = value
        
        # Recent form (last 10 matches)
        home_form_10 = self.calculate_form(home_team, match_date, 10)
        away_form_10 = self.calculate_form(away_team, match_date, 10)
        
        for key, value in home_form_10.items():
            features[f'home_{key}'] = value
        for key, value in away_form_10.items():
            features[f'away_{key}'] = value
        
        # Head-to-head statistics
        h2h = self.calculate_head_to_head(home_team, away_team, match_date)
        features.update(h2h)
        
        # Differential features (home advantage)
        features['goal_diff_differential'] = home_stats['goal_difference'] - away_stats['goal_difference']
        features['points_differential'] = home_stats['points'] - away_stats['points']
        features['form_differential'] = home_form_5['last_5_points'] - away_form_5['last_5_points']
        features['win_rate_differential'] = home_stats['win_rate'] - away_stats['win_rate']
        
        # Add team IDs
        features['home_team_id'] = self.team_to_id.get(home_team, -1)
        features['away_team_id'] = self.team_to_id.get(away_team, -1)
        
        return features
    
    def create_training_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create complete training dataset from historical matches
        
        Returns:
            Tuple of (features_df, target_series)
        """
        print("Creating training dataset...")
        features_list = []
        targets = []
        
        for idx, match in self.matches_df.iterrows():
            # Skip matches without results
            if pd.isna(match['home_score']) or pd.isna(match['away_score']):
                continue
            
            # Create features for this match
            features = self.create_match_features(
                match['home_team'],
                match['away_team'],
                match['date']
            )
            
            # Add match metadata
            features['season'] = match['season']
            features['matchday'] = match['matchday']
            
            features_list.append(features)
            
            # Target: 0 = Away Win, 1 = Draw, 2 = Home Win
            if match['winner'] == 'HOME_TEAM':
                targets.append(2)
            elif match['winner'] == 'AWAY_TEAM':
                targets.append(0)
            else:
                targets.append(1)
        
        features_df = pd.DataFrame(features_list)
        target_series = pd.Series(targets, name='result')
        
        print(f"Created {len(features_df)} training samples with {len(features_df.columns)} features")
        
        return features_df, target_series
    
    def _calculate_streak(self, results: List[str], result_type: str) -> int:
        """Calculate current streak of a specific result type"""
        streak = 0
        for result in reversed(results):
            if result == result_type:
                streak += 1
            else:
                break
        return streak
    
    def _calculate_unbeaten_streak(self, results: List[str]) -> int:
        """Calculate current unbeaten streak (wins + draws)"""
        streak = 0
        for result in reversed(results):
            if result in ['W', 'D']:
                streak += 1
            else:
                break
        return streak
    
    def _get_default_stats(self) -> Dict[str, float]:
        """Return default statistics when no data is available"""
        return {
            'goals_scored': 0,
            'goals_conceded': 0,
            'goal_difference': 0,
            'home_goals_scored': 0,
            'home_goals_conceded': 0,
            'away_goals_scored': 0,
            'away_goals_conceded': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'points': 0,
            'win_rate': 0,
            'draw_rate': 0,
            'loss_rate': 0,
            'avg_goals_scored': 0,
            'avg_goals_conceded': 0,
            'avg_points': 0
        }


if __name__ == '__main__':
    # Example usage
    from data_collector import BundesligaDataCollector
    
    collector = BundesligaDataCollector()
    matches_df = collector.load_data('sample_matches.csv')
    
    if matches_df.empty:
        print("No data found. Run data_collector.py first to generate sample data.")
    else:
        fe = FeatureEngineer(matches_df)
        X, y = fe.create_training_dataset()
        
        print(f"\nDataset shape: {X.shape}")
        print(f"\nTarget distribution:")
        print(f"  Away wins: {(y == 0).sum()}")
        print(f"  Draws: {(y == 1).sum()}")
        print(f"  Home wins: {(y == 2).sum()}")
        print(f"\nSample features:\n{X.head()}")
