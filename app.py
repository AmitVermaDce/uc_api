import os
import uvicorn
from fastapi import FastAPI, Depends
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from uc_api.pipeline.stage_06_sentiment_prediction import SentimentPredictionPipeline, get_sentiment_model
from src.uc_api.model.sentiment.schema import SentimentRequest, SentimentResponse
from src.uc_api.logging import logger

app = FastAPI(
    title="Orange Use Case Predictor",
    description="Sentiment Analysis, Mulitlabel classification and Summary",
    version="1.0.0",
)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/api/v1/train_sentiment_model")
async def sentiment_training():
    try:
        os.system("python sentiment_model_trainer.py")
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


# Predict Sentiment
@app.post(
    "/api/v1/predict_sentiment",
    response_model=SentimentResponse,
)
async def do_sentiment_predict(
    request: SentimentRequest,
    model: SentimentPredictionPipeline = Depends(
        get_sentiment_model,
    ),
):
    try:
        logger.info("<--------Sentiment Model called for prediction-------->")
        sentiment, confidence, probabilities = model.predict_sentiment(
            request.text,
        )        
        logger.info(f"\nText: {request.text} \nSentiment Probabilities: \n\t{probabilities}\nSentiment: {sentiment}\nconfidence: {confidence} \n")
        return SentimentResponse(
            sentiment=sentiment,
            confidence=confidence,
            probabilities=probabilities,
        )
    except Exception as e:
        raise e


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.10", port=8080)
