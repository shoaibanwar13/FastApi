from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
import nltk
from nltk.corpus import wordnet
from fastapi.middleware.cors import CORSMiddleware
import logging
from textblob import TextBlob

# Download necessary NLTK data files
nltk.download('wordnet', quiet=True)
nltk.download('punkt', force=True)
nltk.download('averaged_perceptron_tagger', force=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Define a request model for the input, including heading and text
class ParaphraseRequest(BaseModel):
    text: str
    target_length: int = 100

# Paraphrasing for English text
def get_first_synonym(word, pos=None):
    synonyms = wordnet.synsets(word, pos=pos)
    if synonyms:
        lemma = synonyms[0].lemmas()[0].name()
        if not any(char.isdigit() for char in lemma) and len(lemma) < 20:
            return lemma.replace('_', ' ')
    return word

def paraphrase(text: str) -> str:
    words = nltk.word_tokenize(text)
    paraphrased_text = []

    for word in words:
        pos_tag = nltk.pos_tag([word])[0][1]
        
        if pos_tag.startswith('NN'):
            paraphrased_word = get_first_synonym(word, pos=wordnet.NOUN)
        elif pos_tag.startswith('VB'):
            paraphrased_word = get_first_synonym(word, pos=wordnet.VERB)
        elif pos_tag.startswith('JJ'):
            paraphrased_word = get_first_synonym(word, pos=wordnet.ADJ)
        elif pos_tag.startswith('RB'):
            paraphrased_word = get_first_synonym(word, pos=wordnet.ADV)
        else:
            paraphrased_word = word

        paraphrased_text.append(paraphrased_word)

    paraphrased_sentence = ' '.join(paraphrased_text)
    sentences = nltk.sent_tokenize(paraphrased_sentence)
    capitalized_sentences = [s.capitalize() for s in sentences]
    final_paraphrase = ' '.join(capitalized_sentences)

    # Use TextBlob for grammar correction
    corrected_text = str(TextBlob(final_paraphrase).correct())
    
    return corrected_text

# Function to detect heading and separate it from the paragraph
def detect_heading_and_paragraph(text: str):
    # Split the text into lines
    lines = text.strip().split('\n')
    
    # Assuming the first line is the heading if it is short and does not end with punctuation
    heading = None
    if len(lines) > 0:
        first_line = lines[0].strip()
        if len(first_line.split()) <= 6 and first_line[-1] not in ".!?":
            heading = first_line
            paragraph = '\n'.join(lines[1:]).strip()  # Keep the rest as the paragraph, preserve newlines
        else:
            paragraph = text.strip()
    else:
        paragraph = text.strip()
    
    return heading, paragraph

# API endpoint for text generation/paraphrasing
@app.post("/generate/")
async def generate_text(request: ParaphraseRequest):
    try:
        heading, paragraph = detect_heading_and_paragraph(request.text)

        # General paraphrasing logic for English text
        logger.info(f"Paraphrasing text: {paragraph}")
        paraphrased = paraphrase(paragraph)
        output = f"{heading}\n{paraphrased}" if heading else paraphrased  # Preserve single newline
        return {"generated_text": output}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
