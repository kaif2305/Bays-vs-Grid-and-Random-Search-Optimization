from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import optuna


# Load the dataset
data = load_breast_cancer()
X, y = data.data, data.target

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Training set shape: {X_train.shape} \n Test set shape: {X_test.shape}")


#Train Baseline XGBoost model
baseline_model = XGBClassifier(eval_metric='logloss', random_state=42)
baseline_model.fit(X_train, y_train)

#Evaluate the model
baseline_pred = baseline_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_pred)
print(f"Baseline model accuracy: {baseline_accuracy:.4f}")

#Define the objective function for Optuna
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
    }

    model = XGBClassifier(**params, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    return accuracy

#Creaete the Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

#Best hyperparameters
print("Best hyperparameters: " , study.best_params)
print("Best accuracy: ", study.best_value)

#Define the parameter grid for GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.6, 0.8, 1.0]
}

#Create the GridSearchCV object
grid_search = GridSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42 ),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=1
)

grid_search.fit(X_train, y_train)

#Grid search best parameters and accuracy
print("Grid search best parameter:" , grid_search.best_params_)
print("Grid search best accuracy: ", grid_search.best_score_)

#Define the parameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100, 200, 300, 400],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
}

#Train XGBoost model with RandomizedSearchCV
random_search = RandomizedSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    scoring='accuracy',
    cv=3,
    verbose=1,
    random_state=42
)

#Fit the model
random_search.fit(X_train, y_train)

print("Random search best parameters: ", random_search.best_params_)
print("Random search best accuracy: ", random_search.best_score_)
