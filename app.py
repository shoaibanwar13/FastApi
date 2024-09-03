
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

# Initialize the FastAPI app
app = FastAPI()

# Hugging Face Inference API details
API_URL = "https://api-inference.huggingface.co/models/openai-community/gpt2-medium"
# Replace 'hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' with your actual API token
headers = {"Authorization": "Bearer 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"}

# Define the request body model
class TextRequest(BaseModel):
    inputs: str

# Query function to make request to Hugging Face Inference API
def query(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define the FastAPI endpoint
@app.post("/generate/")
async def generate_text(request: TextRequest):
    # Call the Hugging Face Inference API
    output = query({"inputs": request.inputs})
    return {"generated_text": output}
