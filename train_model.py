import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# -----------------------------
# Load datasets
# -----------------------------
df_encoded = pd.read_csv("data/Cleane_final_Dataset.csv")
df_decoded = pd.read_csv("data/final_worldbank_dataset.csv")

df=pd.read_csv("data/Cleane_final_Dataset.csv")


# Build mapping dictionary
encoded_ids = df_encoded["country"].unique()
country_names = df_decoded["Country"].unique()
id_to_country = dict(zip(encoded_ids, country_names))
country_to_id = {v: k for k, v in id_to_country.items()}

# -----------------------------
# Feature Engineering
# -----------------------------
df_encoded["exports_to_imports_ratio"] = df_encoded["exports_in_billions"] / df_encoded["imports_in_billions"]
df_encoded["gcf_per_capita"] = df_encoded["gcf_in_billions"] / df_encoded["population_in_millions"]
df_encoded["imports_per_capita"] = df_encoded["imports_in_billions"] / df_encoded["population_in_millions"]
df_encoded["exports_per_capita"] = df_encoded["exports_in_billions"] / df_encoded["population_in_millions"]
df_encoded["trade_balance"] = df_encoded["exports_in_billions"] - df_encoded["imports_in_billions"]
df_encoded["fdi_to_gcf_ratio"] = df_encoded["foreign_direct_investment"] / df_encoded["gcf_in_billions"]

df_encoded = df_encoded.dropna()

# -----------------------------
# Modeling with XGBoost
# -----------------------------
drop_features = [
    "fdi_growth", "imports_growth", "exports_growth", "gcf_growth",
    "lagged_gdp_growth", "lagged_fdi", "lagged_exports"
]

X = df_encoded.drop(columns=["gdp_growth_rate"] + [f for f in drop_features if f in df_encoded.columns])
y = df_encoded["gdp_growth_rate"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)
model.fit(X_train, y_train)

# -----------------------------
# Prediction Function
# -----------------------------
def predict(country_name, year, fdi, imports, exports, population, gcf):
    if country_name not in country_to_id:
        raise ValueError(f"Country '{country_name}' not found in mapping.")
    
    encoded_id = country_to_id[country_name]
    
    input_data = pd.DataFrame({
        "country": [encoded_id],
        "year": [year],
        "foreign_direct_investment": [fdi],
        "imports_in_billions": [imports],
        "exports_in_billions": [exports],
        "population_in_millions": [population],
        "gcf_in_billions": [gcf]
    })
    
    # Apply same feature engineering
    input_data["exports_to_imports_ratio"] = input_data["exports_in_billions"] / input_data["imports_in_billions"]
    input_data["gcf_per_capita"] = input_data["gcf_in_billions"] / input_data["population_in_millions"]
    input_data["imports_per_capita"] = input_data["imports_in_billions"] / input_data["population_in_millions"]
    input_data["exports_per_capita"] = input_data["exports_in_billions"] / input_data["population_in_millions"]
    input_data["trade_balance"] = input_data["exports_in_billions"] - input_data["imports_in_billions"]
    input_data["fdi_to_gcf_ratio"] = input_data["foreign_direct_investment"] / input_data["gcf_in_billions"]
    
    prediction = model.predict(input_data)
    return print(prediction[0], {country_name})

print("Predicted GDP Growth Rates for 2025:")

#predict_for_country(
   # country_name="Australia",
    #year=2025,
    #fdi=2.1,                # billions
    #imports=283.6,            # billions
   # exports=343.8,            # billions
  #  population=27.5,        # millions
 #   gcf=490.92                # billions
#)

#predict_for_country(
   # country_name="India",
    #year=2025,
    #fdi=1.2,
    #imports=910,
   # exports=825.3,
  #  population=1460,        # millions
 #   gcf=1445
#)

#predict_for_country(
   # country_name="Germany",
    #year=2025,
   # fdi=0.00223,
    #imports=1540,
   # exports=1765,
  #  population=84.1,          # millions
 #   gcf=1060.0
#)

#predict_for_country(
   # country_name="China",
    #year=2025,
   # fdi=0.055,
    #imports=2583.1,
   # exports=3771.02,
  #  population=1405,        # millions
 #   gcf=8220
#)

#predict_for_country(
 #    country_name="Ireland",
  #   year=2025,
   #  fdi=15.0,
    # imports=600,
     #exports=700,
    # population=5.5,         # millions
    # gcf=120
#)





# -----------------------------
# Save model and mappings
# -----------------------------
model_artifacts = {
    "model": model,
    "country_to_id": country_to_id,
    "id_to_country": id_to_country,
}
with open("gdp_growth_model.pkl", "wb") as f:
     pickle.dump(model_artifacts, f)
     
print("Model saved successfully as gdp_growth_model.pkl")
