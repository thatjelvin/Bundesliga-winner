"""
Machine Learning Models for Bundesliga Match Prediction
Implements Random Forest, XGBoost, and Neural Network models
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any
from config import MODEL_PARAMS, RANDOM_STATE, TEST_SIZE, MODELS_DIR


class BundesligaPredictor:
    """Ensemble predictor for Bundesliga matches"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                     test_size: float = TEST_SIZE) -> Tuple:
        """Prepare and split data for training"""
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """Train Random Forest model"""
        print("Training Random Forest...")
        
        params = MODEL_PARAMS['random_forest']
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        print("Random Forest training completed")
        
        return model
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
        """Train XGBoost model"""
        print("Training XGBoost...")
        
        params = MODEL_PARAMS['xgboost']
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        print("XGBoost training completed")
        
        return model
    
    def train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray) -> keras.Model:
        """Train Neural Network model"""
        print("Training Neural Network...")
        
        params = MODEL_PARAMS['neural_network']
        input_dim = X_train.shape[1]
        
        # Build model
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        
        for units in params['hidden_layers']:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(params['dropout_rate']))
        
        model.add(layers.Dense(3, activation='softmax'))  # 3 classes
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        model.fit(
            X_train, y_train,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_split=0.2,
            verbose=0
        )
        
        self.models['neural_network'] = model
        print("Neural Network training completed")
        
        return model
    
    def train_all(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train all models and return results"""
        print("=" * 60)
        print("Training all models...")
        print("=" * 60)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        results = {}
        
        # Train Random Forest
        rf_model = self.train_random_forest(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        results['random_forest'] = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'predictions': rf_pred
        }
        
        # Train XGBoost
        xgb_model = self.train_xgboost(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        results['xgboost'] = {
            'accuracy': accuracy_score(y_test, xgb_pred),
            'predictions': xgb_pred
        }
        
        # Train Neural Network
        nn_model = self.train_neural_network(X_train, y_train)
        nn_pred = np.argmax(nn_model.predict(X_test), axis=1)
        results['neural_network'] = {
            'accuracy': accuracy_score(y_test, nn_pred),
            'predictions': nn_pred
        }
        
        # Ensemble prediction (voting)
        ensemble_pred = self._ensemble_predict(rf_pred, xgb_pred, nn_pred)
        results['ensemble'] = {
            'accuracy': accuracy_score(y_test, ensemble_pred),
            'predictions': ensemble_pred
        }
        
        # Store test data for evaluation
        results['y_test'] = y_test
        
        print("\n" + "=" * 60)
        print("Training Results:")
        print("=" * 60)
        for model_name, result in results.items():
            if model_name != 'y_test':
                print(f"{model_name.upper()}: {result['accuracy']:.4f}")
        
        return results
    
    def predict(self, X: pd.DataFrame, use_ensemble: bool = True) -> np.ndarray:
        """Make predictions on new data"""
        X_scaled = self.scaler.transform(X)
        
        if use_ensemble:
            # Get predictions from all models
            rf_pred = self.models['random_forest'].predict(X_scaled)
            xgb_pred = self.models['xgboost'].predict(X_scaled)
            nn_pred = np.argmax(self.models['neural_network'].predict(X_scaled), axis=1)
            
            return self._ensemble_predict(rf_pred, xgb_pred, nn_pred)
        else:
            # Use best single model (typically XGBoost or Random Forest)
            return self.models['xgboost'].predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions"""
        X_scaled = self.scaler.transform(X)
        
        # Average probabilities from all models
        rf_proba = self.models['random_forest'].predict_proba(X_scaled)
        xgb_proba = self.models['xgboost'].predict_proba(X_scaled)
        nn_proba = self.models['neural_network'].predict(X_scaled)
        
        ensemble_proba = (rf_proba + xgb_proba + nn_proba) / 3
        
        return ensemble_proba
    
    def _ensemble_predict(self, pred1: np.ndarray, pred2: np.ndarray, 
                         pred3: np.ndarray) -> np.ndarray:
        """Combine predictions using voting"""
        predictions = np.column_stack([pred1, pred2, pred3])
        
        # Majority voting
        ensemble = []
        for preds in predictions:
            values, counts = np.unique(preds, return_counts=True)
            ensemble.append(values[np.argmax(counts)])
        
        return np.array(ensemble)
    
    def save_models(self, prefix: str = 'bundesliga'):
        """Save all models to disk"""
        # Save sklearn models and scaler
        joblib.dump(self.models['random_forest'], MODELS_DIR / f'{prefix}_rf.pkl')
        joblib.dump(self.models['xgboost'], MODELS_DIR / f'{prefix}_xgb.pkl')
        joblib.dump(self.scaler, MODELS_DIR / f'{prefix}_scaler.pkl')
        
        # Save feature names
        joblib.dump(self.feature_names, MODELS_DIR / f'{prefix}_features.pkl')
        
        # Save neural network
        self.models['neural_network'].save(MODELS_DIR / f'{prefix}_nn.h5')
        
        print(f"Models saved to {MODELS_DIR}")
    
    def load_models(self, prefix: str = 'bundesliga'):
        """Load models from disk"""
        self.models['random_forest'] = joblib.load(MODELS_DIR / f'{prefix}_rf.pkl')
        self.models['xgboost'] = joblib.load(MODELS_DIR / f'{prefix}_xgb.pkl')
        self.scaler = joblib.load(MODELS_DIR / f'{prefix}_scaler.pkl')
        self.feature_names = joblib.load(MODELS_DIR / f'{prefix}_features.pkl')
        self.models['neural_network'] = keras.models.load_model(MODELS_DIR / f'{prefix}_nn.h5')
        
        print("Models loaded successfully")
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from Random Forest"""
        if 'random_forest' not in self.models:
            return pd.DataFrame()
        
        importances = self.models['random_forest'].feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        return pd.DataFrame({
            'feature': [self.feature_names[i] for i in indices],
            'importance': importances[indices]
        })


if __name__ == '__main__':
    # Example usage
    from data_collector import generate_sample_data
    from feature_engineering import FeatureEngineer
    
    # Generate sample data
    matches_df = generate_sample_data()
    
    # Create features
    fe = FeatureEngineer(matches_df)
    X, y = fe.create_training_dataset()
    
    # Train models
    predictor = BundesligaPredictor()
    results = predictor.train_all(X, y)
    
    # Save models
    predictor.save_models()
    
    # Show feature importance
    print("\nTop 20 Most Important Features:")
    print(predictor.get_feature_importance())
