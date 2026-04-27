from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI()

# Load the saved model and mappings

with open("gdp_growth_model.pkl", "rb") as f:
    artifacts = pickle.load(f)
    model = artifacts["model"]
    country_to_id = artifacts["country_to_id"]
    
# Define what the input data should look like
class CountryData(BaseModel):
    country_name: str
    year: int
    fdi: float
    imports: float
    exports: float
    population: float
    gcf: float
    
@app.post("/predict")
def get_prediction(data: CountryData):
    if data.country_name not in country_to_id:
        raise
    HTTPException(status_code=404, detail="Country not Found")

# Create the DataFrame for the model 
    input_df = pd.DataFrame({
        "country":[country_to_id[data.country_name]],
        "year": [data.year],
        "foreign_direct_investment": [data.fdi],
        "imports_in_billions": [data.imports],
        "exports_in_billions": [data.exports],
        "population_in_millions": [data.population],
        "gcf_in_billions": [data.gcf]
    
    })


    # Apply same feature engineering
    input_df["exports_to_imports_ratio"] = input_df["exports_in_billions"] / input_df["imports_in_billions"]
    input_df["gcf_per_capita"] = input_df["gcf_in_billions"] / input_df["population_in_millions"]
    input_df["imports_per_capita"] = input_df["imports_in_billions"] / input_df["population_in_millions"]
    input_df["exports_per_capita"] = input_df["exports_in_billions"] / input_df["population_in_millions"]
    input_df["trade_balance"] = input_df["exports_in_billions"] - input_df["imports_in_billions"]
    input_df["fdi_to_gcf_ratio"] = input_df["foreign_direct_investment"] / input_df["gcf_in_billions"]

    prediction = model.predict(input_df)
    return {"country": data.country_name, "predicted_gdp_growth": float(prediction[0])}

