# Regressione lineare

from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

# carica il dataset 
calhou = fetch_california_housing()

# inizializza il modello di regressione lineare
lr = LinearRegression()

# addestra il modello con il dataset 
lr.fit(calhou.data, calhou.target)
print(calhou.data[:5])
print(calhou.target[:5])

# MedInc median income in block group
# HouseAge median house age in block group
# AveRooms average number of rooms per household
# AveBedrms average number of bedrooms per household
# Population block group population
# AveOccup average number of household members
# Latitude block group latitude
# Longitude block group longitude

# target: valore medio in centinaia di migliaia di dollari


# effettua una predizione su un nuovo campione
new_sample = [[ 3.84620000e+00,  5.20000000e+01,  6.28185328e+00,  1.08108108e+00,  5.65000000e+02,  2.18146718e+00,
                 3.78500000e+01, -1.22250000e+02 ]]
predicted_value = lr.predict(new_sample)

print("Valore predetto:", predicted_value)
