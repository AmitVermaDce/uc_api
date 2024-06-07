from typing import Dict
from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    text: str = Field(
        ...,
        example="Testing the model response with some random text",
    )


class SentimentResponse(BaseModel):
    probabilities: Dict[str, float] = Field(
        ...,
        example={
            "negative": 0.02248985692858696,
            "neutral": 0.9616607427597046,
            "positive": 0.005806523375213146,
        },
    )
    sentiment: str = Field(..., example="neutral")
    confidence: float = Field(..., example=96.1660766601562)


class ErrorResponse(BaseModel):
    """
    Error response for the API
    """

    error: bool = Field(..., example=True, title="Whether there is error")
    message: str = Field(..., example="", title="Error message")
    traceback: str = Field(
        None,
        example="",
        title="Detailed traceback of the error",
    )
