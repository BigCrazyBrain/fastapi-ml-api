from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Clase para recibir datos en formato JSON
class InputText(BaseModel):
    texto: str

# Cargar el modelo entrenado
model = joblib.load("modelo_pqr.pkl")

# Crear instancia de FastAPI
app = FastAPI()

@app.post("/clasificar/")
def clasificar_texto(data:InputText):
    texto = data.texto
    prediccion = model.predict([texto])[0]
    return {"Clase_predicha": prediccion}