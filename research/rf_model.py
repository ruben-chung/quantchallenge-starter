import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

def find_data_files():
    """Automatically find the train.csv and test.csv files"""
    possible_paths = [
        ('train.csv', 'test.csv'),  # Current directory
        ('data/train.csv', 'data/test.csv'),  # data subdirectory
        ('research/data/train.csv', 'research/data/test.csv'),  # research/data subdirectory
        ('../data/train.csv', '../data/test.csv'),  # parent/data directory
        ('../research/data/train.csv', '../research/data/test.csv'),  # parent/research/data
    ]
    
    for train_path, test_path in possible_paths:
        if os.path.exists(train_path) and os.path.exists(test_path):
            print(f"✅ Found data files at: {train_path}, {test_path}")
            return train_path, test_path
    
    # If not found, print current directory contents to help debug
    print("❌ Could not find data files. Current directory contents:")
    print(os.listdir('.'))
    
    if os.path.exists('research'):
        print("research/ directory contents:")
        print(os.listdir('research'))
        if os.path.exists('research/data'):
            print("research/data/ directory contents:")
            print(os.listdir('research/data'))
    
    raise FileNotFoundError("Could not find train.csv and test.csv files. Please check your file paths.")

class RandomForestPredictor:
    def __init__(self):
        self.models = {}
        self.feature_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        self.target_names = ['Y1', 'Y2']
        
    def load_data(self, train_path=None, test_path=None):
        """Load training and test datasets with automatic path detection"""
        if train_path is None or test_path is None:
            train_path, test_path = find_data_files()
        
        print(f"Loading data from: {train_path}, {test_path}")
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        print(f"Training data shape: {self.train_df.shape}")
        print(f"Test data shape: {self.test_df.shape}")
        
        # Basic data info
        print(f"\nTraining time range: {self.train_df['time'].min()} to {self.train_df['time'].max()}")
        print(f"Test time range: {self.test_df['time'].min()} to {self.test_df['time'].max()}")
        
        return self
    
    def prepare_features(self, include_time=False, create_interactions=False):
        """Prepare features for training"""
        print("Preparing features...")
        
        # Base features
        features = self.feature_names.copy()
        
        if include_time:
            features = ['time'] + features
            
        # Extract features and targets
        self.X_train = self.train_df[features].copy()
        self.y_train = self.train_df[self.target_names].copy()
        self.X_test = self.test_df[features].copy()
        
        # Optional: Create interaction features
        if create_interactions:
            self.X_train, self.X_test = self._create_interaction_features(self.X_train, self.X_test)
            
        print(f"Feature matrix shape: {self.X_train.shape}")
        print(f"Target matrix shape: {self.y_train.shape}")
        
        return self
    
    def _create_interaction_features(self, X_train, X_test):
        """Create polynomial and interaction features"""
        from sklearn.preprocessing import PolynomialFeatures
        
        print("Creating interaction features...")
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        # Convert back to DataFrame
        feature_names = poly.get_feature_names_out(X_train.columns)
        X_train_new = pd.DataFrame(X_train_poly, columns=feature_names, index=X_train.index)
        X_test_new = pd.DataFrame(X_test_poly, columns=feature_names, index=X_test.index)
        
        print(f"Features expanded from {X_train.shape[1]} to {X_train_new.shape[1]}")
        
        return X_train_new, X_test_new
    
    def train_basic_rf(self, **params):
        """Train basic Random Forest model"""
        print("Training basic Random Forest model...")
        
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(params)
        
        self.models['basic_rf'] = MultiOutputRegressor(
            RandomForestRegressor(**default_params)
        )
        
        self.models['basic_rf'].fit(self.X_train, self.y_train)
        return self
    
    def train_separate_rf(self, **params):
        """Train separate Random Forest models for Y1 and Y2"""
        print("Training separate Random Forest models...")
        
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(params)
        
        # Separate models for Y1 and Y2
        self.models['rf_y1'] = RandomForestRegressor(**default_params)
        self.models['rf_y2'] = RandomForestRegressor(**default_params)
        
        self.models['rf_y1'].fit(self.X_train, self.y_train['Y1'])
        self.models['rf_y2'].fit(self.X_train, self.y_train['Y2'])
        
        return self
    
    def hyperparameter_tuning(self, n_iter=50, cv=3):
        """Perform hyperparameter tuning"""
        print(f"Starting hyperparameter tuning with {n_iter} iterations...")
        
        # Parameter distributions for Random Forest
        param_dist = {
            'estimator__n_estimators': randint(50, 300),
            'estimator__max_depth': [None] + list(randint(5, 30).rvs(10)),
            'estimator__min_samples_split': randint(2, 20),
            'estimator__min_samples_leaf': randint(1, 20),
            'estimator__max_features': ['sqrt', 'log2', None],
            'estimator__bootstrap': [True, False],
            'estimator__max_samples': [None, 0.7, 0.8, 0.9]
        }
        
        # Custom multi-output RMSE scorer
        def multi_output_rmse(y_true, y_pred):
            rmse_y1 = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
            rmse_y2 = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
            return -(rmse_y1 + rmse_y2) / 2
        
        multi_rmse_scorer = make_scorer(multi_output_rmse, greater_is_better=False)
        
        # Create base model
        rf_base = MultiOutputRegressor(RandomForestRegressor(random_state=42, n_jobs=-1))
        
        # Perform randomized search
        start_time = time.time()
        self.rf_random = RandomizedSearchCV(
            rf_base,
            param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring=multi_rmse_scorer,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        # Split data for tuning
        X_train_tune, X_val_tune, y_train_tune, y_val_tune = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42
        )
        
        self.rf_random.fit(X_train_tune, y_train_tune)
        tune_time = time.time() - start_time
        
        print(f"Hyperparameter tuning completed in {tune_time:.2f} seconds")
        print(f"Best score: {-self.rf_random.best_score_:.4f}")
        print(f"Best parameters: {self.rf_random.best_params_}")
        
        # Store the best model
        self.models['tuned_rf'] = self.rf_random.best_estimator_
        
        return self
    
    def evaluate_models(self, test_size=0.2):
        """Evaluate all trained models"""
        print("Evaluating models...")
        
        # Create validation split
        X_train_val, X_val, y_train_val, y_val = train_test_split(
            self.X_train, self.y_train, 
            test_size=test_size, random_state=42
        )
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            if name in ['rf_y1', 'rf_y2']:
                # Handle separate models
                if name == 'rf_y1':
                    model.fit(X_train_val, y_train_val['Y1'])
                    y_pred = model.predict(X_val)
                    
                    rmse = np.sqrt(mean_squared_error(y_val['Y1'], y_pred))
                    mae = mean_absolute_error(y_val['Y1'], y_pred)
                    r2 = r2_score(y_val['Y1'], y_pred)
                    
                    results[name] = {'Y1': {'RMSE': rmse, 'MAE': mae, 'R2': r2}}
                    print(f"  Y1 - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
                
                elif name == 'rf_y2':
                    model.fit(X_train_val, y_train_val['Y2'])
                    y_pred = model.predict(X_val)
                    
                    rmse = np.sqrt(mean_squared_error(y_val['Y2'], y_pred))
                    mae = mean_absolute_error(y_val['Y2'], y_pred)
                    r2 = r2_score(y_val['Y2'], y_pred)
                    
                    results[name] = {'Y2': {'RMSE': rmse, 'MAE': mae, 'R2': r2}}
                    print(f"  Y2 - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            else:
                # Handle multi-output models
                model.fit(X_train_val, y_train_val)
                y_pred = model.predict(X_val)
                
                results[name] = {}
                for i, target in enumerate(self.target_names):
                    y_true_target = y_val.iloc[:, i]
                    y_pred_target = y_pred[:, i]
                    
                    rmse = np.sqrt(mean_squared_error(y_true_target, y_pred_target))
                    mae = mean_absolute_error(y_true_target, y_pred_target)
                    r2 = r2_score(y_true_target, y_pred_target)
                    
                    results[name][target] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
                    print(f"  {target} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        self.validation_results = results
        return results
    
    def predict_test_set(self, model_name='tuned_rf'):
        """Generate predictions for test set"""
        print(f"Generating predictions using {model_name}...")
        
        if model_name not in self.models:
            print(f"Available models: {list(self.models.keys())}")
            model_name = list(self.models.keys())[0]
            print(f"Using {model_name} instead...")
        
        model = self.models[model_name]
        
        # Train on full dataset
        if model_name in ['rf_y1', 'rf_y2']:
            # Handle separate models
            if model_name == 'rf_y1':
                model.fit(self.X_train, self.y_train['Y1'])
                pred_y1 = model.predict(self.X_test)
                # Need Y2 predictions from rf_y2 model
                if 'rf_y2' in self.models:
                    self.models['rf_y2'].fit(self.X_train, self.y_train['Y2'])
                    pred_y2 = self.models['rf_y2'].predict(self.X_test)
                    predictions = np.column_stack([pred_y1, pred_y2])
                else:
                    print("Warning: Y2 model not found. Using basic RF.")
                    return self.predict_test_set('basic_rf')
            else:
                return self.predict_test_set('basic_rf')
        else:
            # Multi-output model
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_test)
        
        # Create prediction dataframe
        pred_df = pd.DataFrame({
            'id': self.test_df['id'],
            'Y1': predictions[:, 0],
            'Y2': predictions[:, 1]
        })
        
        print(f"Predictions generated for {len(pred_df)} test samples")
        print("\nPrediction statistics:")
        print(pred_df[['Y1', 'Y2']].describe())
        
        return pred_df
    
    def feature_importance_analysis(self, model_name='tuned_rf', plot=False):
        """Analyze feature importance"""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        if model_name in ['rf_y1', 'rf_y2']:
            # Single output model
            target = 'Y1' if model_name == 'rf_y1' else 'Y2'
            importances = model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': self.X_train.columns,
                f'{target}_Importance': importances
            }).sort_values(f'{target}_Importance', ascending=False)
            
        else:
            # Multi-output model
            importances_y1 = model.estimators_[0].feature_importances_
            importances_y2 = model.estimators_[1].feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': self.X_train.columns,
                'Y1_Importance': importances_y1,
                'Y2_Importance': importances_y2,
                'Avg_Importance': (importances_y1 + importances_y2) / 2
            }).sort_values('Avg_Importance', ascending=False)
        
        if plot:
            plt.figure(figsize=(12, 8))
            if model_name in ['rf_y1', 'rf_y2']:
                target = 'Y1' if model_name == 'rf_y1' else 'Y2'
                top_features = importance_df.head(15)
                plt.barh(top_features['Feature'], top_features[f'{target}_Importance'])
                plt.title(f'Top 15 Feature Importances - {model_name}')
            else:
                top_features = importance_df.head(15)
                x = np.arange(len(top_features))
                width = 0.35
                
                plt.bar(x - width/2, top_features['Y1_Importance'], width, label='Y1', alpha=0.8)
                plt.bar(x + width/2, top_features['Y2_Importance'], width, label='Y2', alpha=0.8)
                
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.title('Top 15 Feature Importances by Target')
                plt.xticks(x, top_features['Feature'], rotation=45)
                plt.legend()
            
            plt.tight_layout()
            plt.show()
        
        return importance_df

# Main execution
def main():
    print("🌲 Random Forest Model for Y1/Y2 Prediction 🌲")
    print("="*60)
    
    # Initialize predictor
    rf_predictor = RandomForestPredictor()
    
    # Load and prepare data (automatic path detection)
    rf_predictor.load_data()  # Will automatically find the files
    rf_predictor.prepare_features(include_time=False, create_interactions=False)
    
    # Train different Random Forest variants
    print("\n" + "="*60)
    print("TRAINING MULTIPLE RANDOM FOREST MODELS")
    print("="*60)
    
    # 1. Basic Random Forest
    rf_predictor.train_basic_rf(n_estimators=100)
    
    # 2. Separate models for Y1 and Y2
    rf_predictor.train_separate_rf(n_estimators=100)
    
    # 3. Hyperparameter tuned model
    rf_predictor.hyperparameter_tuning(n_iter=30, cv=3)
    
    # Evaluate all models
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    results = rf_predictor.evaluate_models()
    
    # Feature importance analysis
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    importance_df = rf_predictor.feature_importance_analysis('tuned_rf', plot=False)
    print("Top 10 Most Important Features:")
    print(importance_df.head(10))
    
    # Generate final predictions
    print("\n" + "="*60)
    print("GENERATING FINAL PREDICTIONS")
    print("="*60)
    
    # Use the best performing model (usually tuned_rf)
    predictions = rf_predictor.predict_test_set('tuned_rf')
    
    # Save predictions
    predictions.to_csv('random_forest_predictions.csv', index=False)
    print("✅ Predictions saved to 'random_forest_predictions.csv'")
    
    # Save feature importance
    importance_df.to_csv('rf_feature_importance.csv', index=False)
    print("✅ Feature importance saved to 'rf_feature_importance.csv'")
    
    print("\n🎉 Random Forest pipeline completed successfully!")
    return rf_predictor, predictions, importance_df

if __name__ == "__main__":
    rf_predictor, predictions, importance_df = main()