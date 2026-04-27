# GDP Growth Prediction API

End-to-end ML project that predicts GDP growth % using country-level economic indicators. Built with XGBoost and deployed as a FastAPI service for real-time predictions.

**Live Demo:** `POST /predict` → Returns `{"country": "China", "predicted_gdp_growth": 4.00}`

### Screenshots
<img width="600" alt="FastAPI Swagger UI" src="https://github.com/user-attachments/assets/your-image-id">

### Tech Stack
- **ML**: Python, Pandas, Scikit-learn, XGBoost
- **API**: FastAPI, Uvicorn, Pydantic
- **Tools**: VS Code, Git

### Features Used for Prediction
The model was trained on 24 years of panel data with these engineered features:
- `exports_to_imports_ratio`
- `trade_balance` 
- `gcf_per_capita`
- `fdi_to_gcf_ratio`
- `population`
- `capital_formation`

Country names are handled with label encoding for API requests.

### Project Structure
