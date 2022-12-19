import pandas as pd
import numpy as np
from sklearn import model_selection, preprocessing, neural_network, metrics


hit = pd.read_csv("dataset/Hitters.csv")
df = hit.copy()
df = df.dropna()
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
y = df["Salary"]
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=42)


scaler = preprocessing.StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


mlp_model = neural_network.MLPRegressor(hidden_layer_sizes = (100,20)).fit(X_train_scaled, y_train)


print("\n-> Model Detayları : :\n",
    mlp_model
)

print("\n-> Layers :\n",
    mlp_model.n_layers_
)
print("\n-> Hidden Layer Sizes :\n",
    mlp_model.hidden_layer_sizes
)


#! Tahmin
y_pred = mlp_model.predict(X_test_scaled)
np.sqrt(metrics.mean_squared_error(y_test, y_pred))



#! Model Tuning-Geliştirme
mlp_params = {'alpha': [0.1, 0.01, 0.02, 0.005],
             'hidden_layer_sizes': [(20, 20),(100, 50, 150),(300, 200, 150)],
             'activation': ['relu','logistic']}

#En iyi parametrelere sırasıyla verilen ayara göre otomatik karar verir.
mlp_cv_model = model_selection.GridSearchCV(mlp_model, mlp_params, cv = 10)

mlp_cv_model.fit(X_train_scaled, y_train)

print("\n-> En iyi parametreler :\n",
    mlp_cv_model.best_params_
)

mlp_tuned = neural_network.MLPRegressor(alpha = 0.02, hidden_layer_sizes = (100,50,150))
mlp_tuned.fit(X_train_scaled, y_train)
y_pred = mlp_tuned.predict(X_test_scaled)
np.sqrt(metrics.mean_squared_error(y_test, y_pred))

