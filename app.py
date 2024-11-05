from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import requests
import random
from fastapi.middleware.cors import CORSMiddleware

# Ensure nltk resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Hugging Face API setup
API_URL = "https://api-inference.huggingface.co/models/pszemraj/flan-t5-large-grammar-synthesis"
headers = {"Authorization": "Bearer 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"}  # Replace with your token

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Pydantic model for request body
class TextInput(BaseModel):
    text: str

# Helper function to map nltk POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# Get synonyms with context awareness
def get_best_synonym(word, pos):
    synonyms = set()
    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym.lower() != word.lower() and len(synonym.split()) == 1:
                synonyms.add(synonym)
    
    return random.choice(list(synonyms)) if synonyms else word

# Paraphrase function with synonym replacement
def paraphrase_sentence(sentence):
    words = word_tokenize(sentence)
    tagged_words = pos_tag(words)
    
    paraphrased_sentence = []
    
    for word, tag in tagged_words:
        wordnet_pos = get_wordnet_pos(tag)
        if wordnet_pos:
            synonym = get_best_synonym(word, wordnet_pos)
            paraphrased_sentence.append(synonym)
        else:
            paraphrased_sentence.append(word)
    
    return ' '.join(paraphrased_sentence)

# Query the Hugging Face API
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# FastAPI endpoint for text processing
@app.post("/generate/")
async def process_text_with_api(input: TextInput):
    # Paraphrase the text
    paraphrased_text = paraphrase_sentence(input.text)
    
    # Send the paraphrased text to the Hugging Face API for refinement
    response = query({"inputs": paraphrased_text})
    
    # Check and return the refined response from the API
    if "error" not in response:
        refined_text = response[0].get('generated_text', '')
        return {"paraphrased_text": paraphrased_text, "generated_text": refined_text}
    else:
        raise HTTPException(status_code=500, detail=f"API error: {response['error']}")
