"""
Training script for Bundesliga prediction models
Run this to train or retrain models
"""
import sys
from pathlib import Path
from data_collector import BundesligaDataCollector, generate_sample_data
from feature_engineering import FeatureEngineer
from models import BundesligaPredictor


def main():
    print("=" * 70)
    print("Bundesliga Match Prediction Model Training")
    print("=" * 70)
    
    # Step 1: Load or generate data
    print("\n[1/4] Loading data...")
    collector = BundesligaDataCollector()
    matches_df = collector.load_data('sample_matches.csv')
    
    if matches_df.empty:
        print("No existing data found. Generating sample data...")
        matches_df = generate_sample_data()
    
    print(f"Loaded {len(matches_df)} matches")
    
    # Step 2: Feature engineering
    print("\n[2/4] Engineering features...")
    fe = FeatureEngineer(matches_df)
    X, y = fe.create_training_dataset()
    
    print(f"Created dataset with {len(X)} samples and {len(X.columns)} features")
    print(f"Target distribution:")
    print(f"  - Away wins: {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
    print(f"  - Draws: {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")
    print(f"  - Home wins: {(y == 2).sum()} ({(y == 2).sum() / len(y) * 100:.1f}%)")
    
    # Step 3: Train models
    print("\n[3/4] Training models...")
    predictor = BundesligaPredictor()
    results = predictor.train_all(X, y)
    
    # Step 4: Save models
    print("\n[4/4] Saving models...")
    predictor.save_models()
    
    # Display results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    for model_name in ['random_forest', 'xgboost', 'neural_network', 'ensemble']:
        accuracy = results[model_name]['accuracy']
        print(f"{model_name.upper():20s}: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    
    # Feature importance
    print("\n" + "=" * 70)
    print("TOP 15 MOST IMPORTANT FEATURES")
    print("=" * 70)
    importance_df = predictor.get_feature_importance(top_n=15)
    for idx, row in importance_df.iterrows():
        print(f"{row['feature']:40s}: {row['importance']:.4f}")
    
    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("Models saved and ready for predictions")
    print("=" * 70)


if __name__ == '__main__':
    main()
