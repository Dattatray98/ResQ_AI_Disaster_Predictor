from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from src.predict_flood import model
import numpy as np

app = FastAPI()


class prediction_data(BaseModel):
    MonsoonIntensity: float
    RiverManagement: float
    Deforestation: float
    Urbanization: float
    ClimateChange: float
    DamsQuality: float
    Siltation: float
    AgriculturalPractices: float
    Encroachments: float
    IneffectiveDisasterPreparedness: float
    DrainageSystems: float
    CoastalVulnerability: float
    Landslides: float
    Watersheds: float
    DeterioratingInfrastructure: float
    PopulationScore: float
    WetlandLoss: float
    InadequatePlanning: float
    PoliticalFactors: float


@app.post("/predict")
def predict(data: prediction_data):
    data_dict = data.dict()
    formatted_data = {key: [value] for key, value in data_dict.items()}
    input_df = pd.DataFrame(formatted_data)
    pred = model.predict(input_df)
    if isinstance(pred, (np.ndarray, list)):
        pred_value = pred.tolist()
    else:
        pred_value = [float(pred)]

    return {"message": "predicted data", "prediction": pred_value}
