import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class SalesPredictionModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None

    def create_preprocessor(self, categorical_features, numerical_features):
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        numerical_transformer = StandardScaler()

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        return preprocessor

    def build_model(self):
        # Define features
        categorical_features = ['day_of_week', 'month', 'category', 'store_id', 'weather_condition']
        numerical_features = ['avg_price', 'is_weekend', 'is_holiday', 'promotion_active', 'temperature']

        # Create preprocessor
        self.preprocessor = self.create_preprocessor(categorical_features, numerical_features)

        # Create and return the full model pipeline
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_feature_importance(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        feature_names = (self.preprocessor.named_transformers_['num'].get_feature_names_out().tolist() + 
                         self.preprocessor.named_transformers_['cat'].get_feature_names_out().tolist())
        
        importances = self.model.named_steps['regressor'].feature_importances_
        return dict(zip(feature_names, importances))

# Example usage:
# model = SalesPredictionModel()
# model.build_model()
# model.train(X_train, y_train)
# predictions = model.predict(X_test)
# feature_importance = model.get_feature_importance()
