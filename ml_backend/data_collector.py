"""
Bundesliga Data Collection Module
Fetches match data, team statistics, and historical results
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, List, Optional
import time
from config import (
    FOOTBALL_DATA_API_KEY, FOOTBALL_DATA_BASE_URL,
    BUNDESLIGA_ID, DATA_DIR
)


class BundesligaDataCollector:
    """Collects Bundesliga match and team data from various sources"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or FOOTBALL_DATA_API_KEY
        self.base_url = FOOTBALL_DATA_BASE_URL
        self.headers = {'X-Auth-Token': self.api_key}
        self.league_id = BUNDESLIGA_ID
        
    def fetch_matches(self, season: Optional[str] = None, status: str = 'FINISHED') -> pd.DataFrame:
        """
        Fetch matches from the Bundesliga
        
        Args:
            season: Season year (e.g., '2023' for 2023/24 season)
            status: Match status (SCHEDULED, LIVE, IN_PLAY, PAUSED, FINISHED, etc.)
            
        Returns:
            DataFrame with match data
        """
        url = f"{self.base_url}/competitions/{self.league_id}/matches"
        params = {}
        
        if season:
            params['season'] = season
        if status:
            params['status'] = status
            
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            matches = []
            for match in data.get('matches', []):
                matches.append({
                    'match_id': match['id'],
                    'season': match['season']['startDate'][:4],
                    'matchday': match['matchday'],
                    'date': match['utcDate'],
                    'home_team': match['homeTeam']['name'],
                    'home_team_id': match['homeTeam']['id'],
                    'away_team': match['awayTeam']['name'],
                    'away_team_id': match['awayTeam']['id'],
                    'home_score': match['score']['fullTime']['home'],
                    'away_score': match['score']['fullTime']['away'],
                    'status': match['status'],
                    'winner': match['score']['winner']  # HOME_TEAM, AWAY_TEAM, or DRAW
                })
            
            df = pd.DataFrame(matches)
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching matches: {e}")
            return pd.DataFrame()
    
    def fetch_standings(self, season: Optional[str] = None) -> pd.DataFrame:
        """Fetch current Bundesliga standings"""
        url = f"{self.base_url}/competitions/{self.league_id}/standings"
        params = {}
        
        if season:
            params['season'] = season
            
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            standings = []
            for team in data['standings'][0]['table']:
                standings.append({
                    'position': team['position'],
                    'team_name': team['team']['name'],
                    'team_id': team['team']['id'],
                    'played_games': team['playedGames'],
                    'won': team['won'],
                    'draw': team['draw'],
                    'lost': team['lost'],
                    'points': team['points'],
                    'goals_for': team['goalsFor'],
                    'goals_against': team['goalsAgainst'],
                    'goal_difference': team['goalDifference']
                })
            
            return pd.DataFrame(standings)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching standings: {e}")
            return pd.DataFrame()
    
    def fetch_team_info(self, team_id: int) -> Dict:
        """Fetch detailed information about a specific team"""
        url = f"{self.base_url}/teams/{team_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching team info: {e}")
            return {}
    
    def collect_historical_data(self, start_season: int = 2018, end_season: int = 2024) -> pd.DataFrame:
        """
        Collect historical match data for multiple seasons
        
        Args:
            start_season: Starting season year
            end_season: Ending season year (inclusive)
            
        Returns:
            Combined DataFrame with all historical matches
        """
        all_matches = []
        
        for season in range(start_season, end_season + 1):
            print(f"Fetching data for {season}/{season+1} season...")
            matches = self.fetch_matches(season=str(season), status='FINISHED')
            
            if not matches.empty:
                all_matches.append(matches)
            
            # Rate limiting - API typically allows 10 calls per minute
            time.sleep(6)
        
        if all_matches:
            combined_df = pd.concat(all_matches, ignore_index=True)
            self._save_data(combined_df, 'historical_matches.csv')
            return combined_df
        
        return pd.DataFrame()
    
    def collect_current_season_data(self) -> Dict[str, pd.DataFrame]:
        """Collect all relevant data for the current season"""
        print("Collecting current season data...")
        
        current_year = datetime.now().year
        
        # Fetch current season matches
        matches = self.fetch_matches(season=str(current_year))
        time.sleep(6)
        
        # Fetch standings
        standings = self.fetch_standings(season=str(current_year))
        time.sleep(6)
        
        # Fetch upcoming matches
        upcoming = self.fetch_matches(season=str(current_year), status='SCHEDULED')
        
        data = {
            'matches': matches,
            'standings': standings,
            'upcoming': upcoming
        }
        
        # Save data
        for key, df in data.items():
            if not df.empty:
                self._save_data(df, f'current_{key}.csv')
        
        return data
    
    def _save_data(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to CSV"""
        filepath = DATA_DIR / filename
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load data from CSV"""
        filepath = DATA_DIR / filename
        if filepath.exists():
            return pd.read_csv(filepath)
        return pd.DataFrame()


def generate_sample_data():
    """
    Generate sample Bundesliga data for testing when API is not available
    This creates realistic sample data for demonstration purposes
    """
    import numpy as np
    
    teams = [
        'Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen',
        'Union Berlin', 'SC Freiburg', 'Eintracht Frankfurt', 'VfL Wolfsburg',
        'Borussia Mönchengladbach', '1. FC Köln', 'TSG Hoffenheim', 'VfB Stuttgart',
        'FC Augsburg', 'Werder Bremen', 'VfL Bochum', 'FSV Mainz 05',
        'Hertha BSC', 'FC Schalke 04'
    ]
    
    matches = []
    match_id = 1
    
    # Generate 3 seasons of data
    for season in range(2021, 2024):
        # Each team plays each other twice (home and away)
        for matchday in range(1, 35):
            # Generate random matches
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            
            # Simulate match outcome with some realism
            home_advantage = 0.3
            home_strength = 0.5 + (0.5 if home_team in ['Bayern Munich', 'Borussia Dortmund'] else 0)
            
            home_score = np.random.poisson(1.5 + home_advantage + home_strength * 0.5)
            away_score = np.random.poisson(1.2)
            
            if home_score > away_score:
                winner = 'HOME_TEAM'
            elif away_score > home_score:
                winner = 'AWAY_TEAM'
            else:
                winner = 'DRAW'
            
            match_date = datetime(season, 8, 1) + timedelta(days=matchday * 7)
            
            matches.append({
                'match_id': match_id,
                'season': season,
                'matchday': matchday,
                'date': match_date.isoformat(),
                'home_team': home_team,
                'home_team_id': teams.index(home_team) + 1,
                'away_team': away_team,
                'away_team_id': teams.index(away_team) + 1,
                'home_score': home_score,
                'away_score': away_score,
                'status': 'FINISHED',
                'winner': winner
            })
            
            match_id += 1
    
    df = pd.DataFrame(matches)
    
    # Save sample data
    filepath = DATA_DIR / 'sample_matches.csv'
    df.to_csv(filepath, index=False)
    print(f"Sample data generated and saved to {filepath}")
    
    return df


if __name__ == '__main__':
    # Example usage
    collector = BundesligaDataCollector()
    
    # Try to fetch real data, fall back to sample data if API key not configured
    if FOOTBALL_DATA_API_KEY and FOOTBALL_DATA_API_KEY != 'YOUR_API_KEY_HERE':
        print("Collecting real data from API...")
        data = collector.collect_current_season_data()
    else:
        print("API key not configured. Generating sample data...")
        generate_sample_data()
