from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
import joblib
import pandas as pd

app = FastAPI(
    title="Deploy Depression Tanisha Miranda",
    version="0.0.1"
)

# Cargar el modelo entrenado
model_rf = joblib.load('modelo_depresion_rf_v01.pkl')


# Endpoint para realizar predicciones con parámetros de consulta
@app.post("/api/v1/predict-depression")
async def predict(
        sex: float,
        age: float,
        married: float,
        number_children: float,
        total_members: float,
        incoming_salary: float
):
    data = {
        'sex': sex,
        'age': age,
        'married': married,
        'number_children': number_children,
        'total_members': total_members,
        'incoming_salary': incoming_salary
    }

    try:
        df_new_data = pd.DataFrame([data])
        prediction = model_rf.predict(df_new_data)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"prediction": prediction.tolist()}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
