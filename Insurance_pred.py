import kagglehub
import pandas as pd
import os



# Download latest version
path = kagglehub.dataset_download("harishkumardatalab/medical-insurance-price-prediction")
# Salviamo il dataset in insurance
insurance = pd.read_csv(path + '/Medical_insurance.csv')

# print(insurance.info())
# print(insurance.head())

import matplotlib.pyplot as plt

plt.rc('font', size=8)
plt.rc('axes', labelsize=8, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=10)

# rimuoviamo i duplicati
insurance = insurance.drop_duplicates()


# print(insurance.head())
# print(insurance["region"].value_counts())



corr_matrix = insurance.corr(numeric_only= True)

# print(corr_matrix)

from pandas.plotting import scatter_matrix

attributes = ["bmi" , "age", "children", "charges"]
scatter_matrix(insurance[attributes], figsize=(12,8))

# plt.show()


# analisi dei valori comparati con charges

# smoker = insurance.groupby("smoker")["charges"].mean()
# age = insurance.groupby("sex")["charges"].mean()
# region = insurance.groupby("region")["charges"].mean()

# analisi dei valori presenti nelle categorie

# print(insurance["sex"].value_counts())
# print(insurance["smoker"].value_counts())
# print(insurance["region"].value_counts())

from sklearn.model_selection import train_test_split

# split stratificato con smoker
train_insurance , test_insurance = train_test_split( 
    insurance , 
    test_size= 0.15 , 
    random_state  = 42 , 
    stratify= insurance["smoker"]
    )

# verifica delle proporzioni di smoker nel test e train set

# print(train_insurance["smoker"].value_counts())
# print(test_insurance["smoker"].value_counts())

insurance = train_insurance.copy()

# creazione di un nuovo dataset con solo il target (charges) e uno con solo i dati 
insurance_lab = insurance["charges"].copy()
insurance_feat = insurance.drop("charges", axis = 1)

import numpy as np

# divido il set in elementi categorici e numerici
insurance_num = insurance.select_dtypes(include = [np.number])
insurance_cat = insurance.select_dtypes(include = ["object" , "string"])



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

num = ["age" , "bmi", "children"]
cat = ["region", "smoker", "sex"]
# usiamo OneHotEncoder per le categorie testuali e StandardScaler per quelle numeriche

# creiamo il preprocessing che transformi automaticamente le colonne in base al tipo di dato ( categorico o numerico)
preprocessing = ColumnTransformer([

    ("num" , StandardScaler() , num)

    ,("cat" , OneHotEncoder() , cat)])

# usando il columntransformer il nostro modello andrà a cercare queste colonne e applicare le trasformazioni


from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# creiamo una pipeline per semplificare il processo con il modello di regressione lineare
lin_reg = make_pipeline( preprocessing , LinearRegression())

lin_reg.fit(insurance_feat , insurance_lab)

insurance_predictions = lin_reg.predict(insurance_feat)

from sklearn.metrics import root_mean_squared_error

lin_rmse = root_mean_squared_error(insurance_lab , insurance_predictions )

from sklearn.model_selection import cross_val_score

lin_cross = -cross_val_score(lin_reg , insurance_feat , insurance_lab ,scoring = "neg_root_mean_squared_error", cv = 5)

# LinearRegression
# training RMSE ≈ 6210
# CV RMSE ≈ 6253
# Il modello probabilmente è troppo semplice, passiamo quindi a forest senza cross_validation


from sklearn.ensemble import RandomForestRegressor 

forest_reg = make_pipeline(preprocessing , RandomForestRegressor(random_state= 42))

forest_reg.fit(insurance_feat , insurance_lab)

insurance_predictions =  forest_reg.predict(insurance_feat)

forest_rmse = root_mean_squared_error(insurance_lab , insurance_predictions)

#print(forest_rmse) 1892.979402225559 (chiaro overfitting al training)

forest_cross = -cross_val_score(forest_reg , insurance_feat, insurance_lab , cv = 5 , scoring = "neg_root_mean_squared_error")

from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

grid_params = { 
    "randomforestregressor__max_depth" : randint(low =2 , high = 15),
    "randomforestregressor__min_samples_leaf" : randint(low = 1 , high  = 50),
    "randomforestregressor__n_estimators" : randint(low = 50 , high  = 500) 
}


rnd_search = RandomizedSearchCV( forest_reg , param_distributions= grid_params , n_iter= 10 , scoring = "neg_root_mean_squared_error" , cv = 5 , verbose = 1 , random_state= 42)

rnd_search.fit(insurance_feat , insurance_lab)

final_model = rnd_search.best_estimator_

# print(rnd_search.best_params_) {'randomforestregressor__max_depth': 12, 'randomforestregressor__min_samples_leaf': 8, 'randomforestregressor__n_estimators': 238}
#print(-rnd_search.best_score_) 4702.572311472615 (un miglioramento di 300 dollari su CV)

final_predictions = final_model.predict(insurance_feat)

final_rmse = root_mean_squared_error(insurance_lab , final_predictions)

#   print(final_rmse) 4035.1025236958426 (la differenza fra training e CV è diminuita, quindi anche l'overfitting)
# proviamo usando il log e ritrasformando successivamente i risultati 
""""
from sklearn.compose import TransformedTargetRegressor

log_model = TransformedTargetRegressor(
    regressor= final_model,
    func= np.log,
    inverse_func= np.exp
)

log_model.fit(insurance_feat , insurance_lab)

log_predictions = log_model.predict(insurance_feat)

log_rmse = root_mean_squared_error(insurance_lab , log_predictions)
# print(log_rmse) 4258.06624912132 (il modello performa peggio sul training set formato senza CV)
log_cross = -cross_val_score(log_model , insurance_feat, insurance_lab , cv = 5 , scoring = "neg_root_mean_squared_error")
"""
# print(pd.Series(log_cross).describe()) mean  4680 (il modello migliora molto sensibilmente)
# per trainare il modello finale teniamo solamente il modello tuned senza log


X_test = test_insurance.drop("charges", axis=1)
y_test = test_insurance["charges"].copy()
final_predictions = final_model.predict(X_test)
final_rmse = root_mean_squared_error(y_test, final_predictions)


# print(final_rmse) 3331.6623267981586

my_data = pd.DataFrame([
    {
        "age": 20,
        "sex": "male",
        "bmi": 23.5,
        "children": 0,
        "smoker": "no",
        "region": "northeast"
    }
])


my_prediction = final_model.predict(my_data)

print(my_prediction)