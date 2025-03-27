from fastapi import FastAPI
import uvicorn
import os
import sys
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from textSummarizer.pipeline.prediction import PredictionPipeline

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response(content="Training successful!!", media_type="text/plain")
    except Exception as e:
        return Response(content=f"Error Occurred! {e}", media_type="text/plain")


@app.post("/predict")
async def predict_route(text: str):
    try:
        obj = PredictionPipeline()
        result = obj.predict(text)
        return {"summary": result}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
