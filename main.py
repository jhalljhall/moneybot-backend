from starlette.middleware.cors import CORSMiddleware

from app.api.api_v1.api import api_router
from app.core.config import settings

from typing import  AsyncGenerator, Any, List, Annotated, NoReturn

from fastapi import FastAPI,APIRouter, Body, Depends, HTTPException, Form, Query, WebSocket
from fastapi.encoders import jsonable_encoder
from pydantic.networks import EmailStr
from sqlalchemy.orm import Session

from app import controllers, models, schemas
from app.api import deps
from app.core.config import settings
from app.utils import send_new_account_email
from datetime import timedelta
from app.core import security
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import openai
import os
import sys
import json
from starlette import status

from openai import AsyncOpenAI

app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set all CORS enabled origins
# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],


async def get_ai_response(message: str) -> AsyncGenerator[str, None]:
    """
    OpenAI Response
    """
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant, skilled in explaining "
                    "complex concepts in simple terms."
                ),
            },
            {
                "role": "user",
                "content": message,
            },
        ],
        stream=True,
    )

    all_content = ""
    async for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            all_content += content
            yield all_content

@app.get("/")
async def root():
    return {"message":"Hello World"}

app.include_router(api_router, prefix=settings.API_V1_STR)