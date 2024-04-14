from typing import Any, List, Annotated, NoReturn

from fastapi import APIRouter, Body, Depends, HTTPException, Form, Query, WebSocket
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

# Getting OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("Please set OPENAI_API_KEY environment variable. Exiting.")
    sys.exit(1)

# Parameters for OpenAI
openai_model = "gpt-3.5-turbo"
max_responses = 1
temperature = 0.7
max_tokens = 512

# Initialize OpenAI client
#client = openai.OpenAI()
client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
# Defining error in case of 503 from OpenAI
error503 = "OpenAI server is busy, try again later"

class ChatModelSchema(BaseModel):
    message: str

router = APIRouter()
@router.post("/prompt")
def get_response_openai(prompt: str = Query(..., description="The prompt for the OpenAI model")):
    
    try:
        response = client.chat.completions.create(
            model=openai_model,
            temperature=temperature,
            max_tokens=max_tokens,
            n=max_responses,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            messages=[
                {"role": "system", "content": "You are an expert financial educator. You give advice to Users."},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
    except Exception as e:
        print(f"Error in creating financial educator from OpenAI: {str(e)}")
        raise HTTPException(status_code=503, detail=error503)
    
    try:
        for chunk in response:
            # Check if delta exists and if 'content' is present
           
            if 'delta' in chunk.choices[0] and 'content' in chunk.choices[0].delta:
                yield chunk.choices[0].delta['content']

    except Exception as e:
        print(f"OpenAI Response (Streaming) Error: {str(e)}")
        raise HTTPException(status_code=503, detail=error503)

# This assumes more code follows for the full implementation of the FastAPI app

def get_openai_generator(question: str):
    
    openai_stream = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are an expert financial educator. You give advice to Users."},
                {"role": "user", "content": question}
            ],
        temperature=0.0,
        stream=True,
    )
    
    for event in openai_stream:
        if event.choices[0].delta.content is not None:
            # yield "data: " + event.choices[0].delta.content + "\n\n"
            
            yield f"{json.dumps({'message': event.choices[0].delta.content})}_"

@router.post("/chat_model")
async def chat_model(s: ChatModelSchema):
    # question = "What are the advantages of gemma?"    
    try:
        
        return StreamingResponse(get_openai_generator(s.message)
                                 , media_type='text/event-stream')
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
