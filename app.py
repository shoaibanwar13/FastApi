from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialize the FastAPI app
app = FastAPI()

# Load the text generation pipeline
pipe = pipeline("text2text-generation", model="Rutts07/t5-ai-human-gen")

# Define a request body model
class TextRequest(BaseModel):
    text: str

# Define the endpoint
@app.post("/generate/")
async def generate_text(request: TextRequest):
    # Use the pipeline to generate text
    generated_text = pipe(
        request.text,
    )[0]['generated_text' ]
    return {"generated_text": generated_text}

# To run the application, use the command:
# uvicorn main:app --reload
