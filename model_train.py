from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Función para entrenar un modelo de Árbol de Decisión
def train_decision_tree(df, target_col, feature_cols):
    X = df[feature_cols]
    y = df[target_col]
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar el modelo
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred)**0.5,  # RMSE calculado manualmente
        "R2": r2_score(y_test, y_pred)
    }
    
    return model, metrics

# Función para entrenar un modelo de Random Forest
def train_random_forest(df, target_col, feature_cols):
    X = df[feature_cols]
    y = df[target_col]
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar el modelo
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred)**0.5,  # RMSE calculado manualmente
        "R2": r2_score(y_test, y_pred)
    }
    
    return model, metrics

