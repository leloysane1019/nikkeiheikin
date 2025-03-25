from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import numpy as np
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = load_model("nikkei_model.keras")

@app.get("/", response_class=HTMLResponse)
def form_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict_web", response_class=HTMLResponse)
def predict_web(request: Request, index: str = Form(...)):
    data = yf.download(index, period="180d")["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X = scaled[-30:].reshape(1, 30, 1)
    pred = model.predict(X)
    result = round(float(scaler.inverse_transform(pred)[0][0]), 2)
    return templates.TemplateResponse("index.html", {"request": request, "result": result})
