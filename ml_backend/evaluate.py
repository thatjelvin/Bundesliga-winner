"""
Model Evaluation and Visualization
Generates detailed performance metrics and visualizations
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import joblib
from pathlib import Path

from data_collector import BundesligaDataCollector, generate_sample_data
from feature_engineering import FeatureEngineer
from models import BundesligaPredictor
from config import MODELS_DIR, DATA_DIR

# Set style for better-looking plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class ModelEvaluator:
    """Evaluate and visualize model performance"""
    
    def __init__(self, predictor: BundesligaPredictor, X_test: np.ndarray, y_test: np.ndarray):
        self.predictor = predictor
        self.X_test = X_test
        self.y_test = y_test
        self.class_names = ['Away Win', 'Draw', 'Home Win']
        
    def evaluate_all_models(self) -> pd.DataFrame:
        """Evaluate all models and return metrics DataFrame"""
        results = []
        
        for model_name in ['random_forest', 'xgboost', 'neural_network']:
            if model_name == 'neural_network':
                y_pred = np.argmax(self.predictor.models[model_name].predict(self.X_test), axis=1)
                y_proba = self.predictor.models[model_name].predict(self.X_test)
            else:
                y_pred = self.predictor.models[model_name].predict(self.X_test)
                y_proba = self.predictor.models[model_name].predict_proba(self.X_test)
            
            metrics = self._calculate_metrics(y_pred, y_proba, model_name)
            results.append(metrics)
        
        # Ensemble
        ensemble_pred = self.predictor.predict(pd.DataFrame(self.X_test, columns=self.predictor.feature_names))
        ensemble_proba = self.predictor.predict_proba(pd.DataFrame(self.X_test, columns=self.predictor.feature_names))
        metrics = self._calculate_metrics(ensemble_pred, ensemble_proba, 'ensemble')
        results.append(metrics)
        
        return pd.DataFrame(results)
    
    def _calculate_metrics(self, y_pred: np.ndarray, y_proba: np.ndarray, model_name: str) -> dict:
        """Calculate comprehensive metrics for a model"""
        return {
            'model': model_name,
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision_macro': precision_score(self.y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(self.y_test, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(self.y_test, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        }
    
    def plot_confusion_matrices(self, save_path: str = None):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Confusion Matrices - Bundesliga Match Prediction', fontsize=16, fontweight='bold')
        
        model_names = ['random_forest', 'xgboost', 'neural_network', 'ensemble']
        titles = ['Random Forest', 'XGBoost', 'Neural Network', 'Ensemble']
        
        for idx, (model_name, title) in enumerate(zip(model_names, titles)):
            ax = axes[idx // 2, idx % 2]
            
            if model_name == 'ensemble':
                y_pred = self.predictor.predict(pd.DataFrame(self.X_test, columns=self.predictor.feature_names))
            elif model_name == 'neural_network':
                y_pred = np.argmax(self.predictor.models[model_name].predict(self.X_test), axis=1)
            else:
                y_pred = self.predictor.models[model_name].predict(self.X_test)
            
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=self.class_names, yticklabels=self.class_names)
            ax.set_title(f'{title}\nAccuracy: {accuracy_score(self.y_test, y_pred):.4f}')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrices saved to {save_path}")
        
        return fig
    
    def plot_feature_importance(self, top_n: int = 20, save_path: str = None):
        """Plot feature importance from Random Forest"""
        importance_df = self.predictor.get_feature_importance(top_n=top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
        bars = ax.barh(range(len(importance_df)), importance_df['importance'], color=colors)
        
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('Importance', fontweight='bold', fontsize=12)
        ax.set_title(f'Top {top_n} Most Important Features (Random Forest)', 
                    fontweight='bold', fontsize=14)
        ax.invert_yaxis()
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, importance_df['importance'])):
            ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        return fig
    
    def plot_model_comparison(self, metrics_df: pd.DataFrame, save_path: str = None):
        """Plot comparison of all models"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        ax1 = axes[0]
        models = metrics_df['model'].str.replace('_', ' ').str.title()
        accuracies = metrics_df['accuracy']
        
        colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
        bars = ax1.bar(range(len(models)), accuracies, color=colors)
        
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=15, ha='right')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('Model Accuracy Comparison', fontweight='bold', fontsize=14)
        ax1.set_ylim([0, 1])
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # F1 Score comparison
        ax2 = axes[1]
        metrics_subset = metrics_df[['model', 'precision_macro', 'recall_macro', 'f1_macro']]
        metrics_subset = metrics_subset.set_index('model')
        
        x = np.arange(len(metrics_subset))
        width = 0.25
        
        ax2.bar(x - width, metrics_subset['precision_macro'], width, label='Precision', color='#4CAF50')
        ax2.bar(x, metrics_subset['recall_macro'], width, label='Recall', color='#2196F3')
        ax2.bar(x + width, metrics_subset['f1_macro'], width, label='F1 Score', color='#FF9800')
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_subset.index.str.replace('_', ' ').str.title(), rotation=15, ha='right')
        ax2.set_ylabel('Score', fontweight='bold')
        ax2.set_title('Precision, Recall, and F1 Score Comparison', fontweight='bold', fontsize=14)
        ax2.set_ylim([0, 1])
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        
        return fig
    
    def plot_class_performance(self, save_path: str = None):
        """Plot per-class performance for the ensemble model"""
        y_pred = self.predictor.predict(pd.DataFrame(self.X_test, columns=self.predictor.feature_names))
        
        # Calculate per-class metrics
        precision_per_class = precision_score(self.y_test, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(self.y_test, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(self.y_test, y_pred, average=None, zero_division=0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        ax.bar(x - width, precision_per_class, width, label='Precision', color='#4CAF50')
        ax.bar(x, recall_per_class, width, label='Recall', color='#2196F3')
        ax.bar(x + width, f1_per_class, width, label='F1 Score', color='#FF9800')
        
        ax.set_xlabel('Class', fontweight='bold', fontsize=12)
        ax.set_ylabel('Score', fontweight='bold', fontsize=12)
        ax.set_title('Per-Class Performance (Ensemble Model)', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class performance plot saved to {save_path}")
        
        return fig
    
    def generate_report(self, save_path: str = None):
        """Generate comprehensive evaluation report"""
        print("=" * 70)
        print("BUNDESLIGA MATCH PREDICTION - MODEL EVALUATION REPORT")
        print("=" * 70)
        
        # Overall metrics
        metrics_df = self.evaluate_all_models()
        print("\n1. OVERALL MODEL PERFORMANCE")
        print("-" * 70)
        print(metrics_df.to_string(index=False))
        
        # Per-class performance for ensemble
        print("\n2. PER-CLASS PERFORMANCE (ENSEMBLE MODEL)")
        print("-" * 70)
        y_pred = self.predictor.predict(pd.DataFrame(self.X_test, columns=self.predictor.feature_names))
        report = classification_report(self.y_test, y_pred, target_names=self.class_names)
        print(report)
        
        # Feature importance
        print("\n3. TOP 15 MOST IMPORTANT FEATURES")
        print("-" * 70)
        importance_df = self.predictor.get_feature_importance(top_n=15)
        print(importance_df.to_string(index=False))
        
        print("\n" + "=" * 70)
        
        # Save report to file
        if save_path:
            with open(save_path, 'w') as f:
                f.write("=" * 70 + "\n")
                f.write("BUNDESLIGA MATCH PREDICTION - MODEL EVALUATION REPORT\n")
                f.write("=" * 70 + "\n\n")
                f.write("1. OVERALL MODEL PERFORMANCE\n")
                f.write("-" * 70 + "\n")
                f.write(metrics_df.to_string(index=False) + "\n\n")
                f.write("2. PER-CLASS PERFORMANCE (ENSEMBLE MODEL)\n")
                f.write("-" * 70 + "\n")
                f.write(report + "\n\n")
                f.write("3. TOP 15 MOST IMPORTANT FEATURES\n")
                f.write("-" * 70 + "\n")
                f.write(importance_df.to_string(index=False) + "\n")
            
            print(f"\nReport saved to {save_path}")


def main():
    """Main evaluation function"""
    print("Loading data and models...")
    
    # Load data
    collector = BundesligaDataCollector()
    matches_df = collector.load_data('sample_matches.csv')
    
    if matches_df.empty:
        print("Generating sample data...")
        matches_df = generate_sample_data()
    
    # Create features
    fe = FeatureEngineer(matches_df)
    X, y = fe.create_training_dataset()
    
    # Load trained models
    predictor = BundesligaPredictor()
    predictor.load_models()
    
    # Prepare test data
    X_train, X_test, y_train, y_test = predictor.prepare_data(X, y)
    
    # Create evaluator
    evaluator = ModelEvaluator(predictor, X_test, y_test)
    
    # Generate all visualizations
    output_dir = DATA_DIR / 'evaluation'
    output_dir.mkdir(exist_ok=True)
    
    print("\nGenerating evaluation visualizations...")
    
    evaluator.plot_confusion_matrices(save_path=output_dir / 'confusion_matrices.png')
    evaluator.plot_feature_importance(save_path=output_dir / 'feature_importance.png')
    
    metrics_df = evaluator.evaluate_all_models()
    evaluator.plot_model_comparison(metrics_df, save_path=output_dir / 'model_comparison.png')
    evaluator.plot_class_performance(save_path=output_dir / 'class_performance.png')
    
    # Generate text report
    evaluator.generate_report(save_path=output_dir / 'evaluation_report.txt')
    
    print("\nEvaluation complete! All results saved to:", output_dir)


if __name__ == '__main__':
    main()
