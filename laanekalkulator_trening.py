# Importering av pakker
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import time
import joblib
from dataanalyse_og_preprosessering import data_analysis

# Henter variabler fra funksjonen data_analysis() i filen dataanalyse_og_preprosessering.py
train, test, train_encoded, test_encoded, x_train, y_train, X_train, X_val, y_val = data_analysis()

time_durations = {}

start_time = time.time()
# Definerer parameter-rutenettet som skal søkes gjennom
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None]
}

# Opprett en Random Forest klassifiserer
rf_classifier = RandomForestClassifier(random_state=42)

# Oppretter og kjører GridSearchCV-objektet
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print de beste hyperparameterene
print("Best Hyperparameters:", grid_search.best_params_)

# Hent ut den beste modellen
best_rf_model = grid_search.best_estimator_

# Evaluerer modellen på valideringssettet
y_val_pred = best_rf_model.predict(X_val)
f1 = f1_score(y_val, y_val_pred)
print("Validation f1 score:", f1)

end_time = time.time()
time_durations["Training"] = end_time - start_time


# Eksportér den trente modellen til en fil
model_filename = 'trained_model.joblib'
joblib.dump(best_rf_model, model_filename)




print("Ferdig!")

for key,value in time_durations.items():
    print(key, ":", round(value,2), "sek")
