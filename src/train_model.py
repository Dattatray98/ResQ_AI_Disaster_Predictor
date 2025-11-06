import pandas as pd    # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.neighbors import KNeighborsRegressor # type: ignore
from sklearn.metrics import r2_score, mean_squared_error # type: ignore
import joblib # type: ignore

df = pd.read_csv("../data/processed/flood.csv")

X = df[['MonsoonIntensity','TopographyDrainage','RiverManagement','Deforestation','Urbanization','ClimateChange','DamsQuality','Siltation','AgriculturalPractices','Encroachments','IneffectiveDisasterPreparedness','DrainageSystems','CoastalVulnerability','Landslides','Watersheds','DeterioratingInfrastructure','PopulationScore','WetlandLoss','InadequatePlanning','PoliticalFactors']]

y = df[["FloodProbability"]]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mod = KNeighborsRegressor()

mod.fit(X_train, y_train)

joblib.dump(mod, '../models/flood_model.pkl')

print("training completed model saved as flood_model.pkl")