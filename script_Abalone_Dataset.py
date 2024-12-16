import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import statistics

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn import metrics


import os
os.chdir("c:/Users/HUAWEI/Desktop/ML_2022/Machine learning/Keagles")

ListaZIP = list(filter(lambda f : f.endswith(".zip"), os.listdir()))

from zipfile import ZipFile
trainData = ZipFile(ListaZIP[0], "r") #Queremos solo el unico elemento, o puedo usar for si necesito ciertos elementos del zip
trainData.extractall(".") #Para descomprimir en el actual escritorio es un punto
trainData.close()


# %% Preproc
train = pd.read_csv("train.csv")
train = train.drop_duplicates()
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

sample_submission =  sample_submission.iloc[:, 1]
sample_submission = sample_submission.astype("float64").values

train.isnull().sum().sort_values(ascending = False)
test.isnull().sum().sort_values(ascending = False)

# Para entrenamiento
# train.info()
# train.id.unique()

# train.Rings.unique()

# One hot encoding
for col in train.dtypes[train.dtypes == "object"].index:
    Col4Dummy = train.pop(col)
    train = pd.concat([train, pd.get_dummies(Col4Dummy, prefix = col)], axis =1)
    
train.columns

# Para testeo


# train.Rings.unique()

# One hot encoding
for col in test.dtypes[test.dtypes == "object"].index:
    Col4Dummy_test = test.pop(col)
    test = pd.concat([test, pd.get_dummies(Col4Dummy_test, prefix = col)], axis =1)



train.rename(columns = {"Whole weight": "Peso_total",
                                "Whole weight.1" : "Peso_total_1",
                                "Whole weight.2" : "Peso_total_2",
                                "Shell weight" : "Peso_carcasa"}, inplace = True)

test.rename(columns = {"Whole weight": "Peso_total",
                                "Whole weight.1" : "Peso_total_1",
                                "Whole weight.2" : "Peso_total_2",
                                "Shell weight" : "Peso_carcasa"}, inplace = True)



# Correlacion




train.columns
##################
#Elimino sex_M, Peso 2 y diametro
train = train.drop(["Sex_M","Diameter","Peso_total_2"] , axis = 1)
test = test.drop(["Sex_M","Diameter","Peso_total_2"] , axis = 1)


train.Peso_total.sort_values()

# %% Modelado
# help(train.iloc)

X = train.iloc[:, 1:] 
X = X.drop("Rings", axis =1)

y = train.Rings 

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.15,
                                                    stratify=y)


#######################
RFBase = RandomForestRegressor(n_estimators= 544,
                               min_samples_leaf=60,
                               max_depth=8,
                               min_samples_split=13)

RFBase.fit(X_train, y_train)
RFBase.score(X_train, y_train)

# %% Prediccion

y_model = RFBase.predict(X_test)

# Eliminar Sex_F
# Eliminar las correlaciones Diametro y Peso 2
# train test split de los train

y_test = y_test.astype("float64")


#ListaRoot = []



#statistics.mean(ListaRoot)
#################
# %% 
X_test_pred = test.iloc[:, 1:]
y_pred = RFBase.predict(X_test_pred)


np.sqrt(metrics.mean_squared_log_error(sample_submission, y_pred))

#0.2477625021282989






    
# %%
