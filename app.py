from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
from nltk.corpus import wordnet
from fastapi.middleware.cors import CORSMiddleware

# Download necessary NLTK data files
nltk.download('wordnet', quiet=True)
nltk.download('punkt', force=True)
nltk.download('averaged_perceptron_tagger', force=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Define a request model for the input
class ParaphraseRequest(BaseModel):
    text: str

def get_first_synonym(word, pos=None):
    """Retrieve the first synonym from WordNet if available and appropriate."""
    synonyms = wordnet.synsets(word, pos=pos)
    if synonyms:
        # Return the first lemma as the synonym, unless it's too technical or awkward
        lemma = synonyms[0].lemmas()[0].name()
        # Ensure the lemma doesn't contain numbers, acronyms, or technical terms
        if not any(char.isdigit() for char in lemma) and len(lemma) < 20:
            return lemma.replace('_', ' ')
    return word  # Return the original word if no suitable synonym is found

def paraphrase(text: str) -> str:
    # Tokenize the input text into words
    words = nltk.word_tokenize(text)
    paraphrased_text = []

    for word in words:
        pos_tag = nltk.pos_tag([word])[0][1]
        
        # Only replace nouns, verbs, adjectives, and adverbs
        if pos_tag.startswith('NN'):
            paraphrased_word = get_first_synonym(word, pos=wordnet.NOUN)
        elif pos_tag.startswith('VB'):
            paraphrased_word = get_first_synonym(word, pos=wordnet.VERB)
        elif pos_tag.startswith('JJ'):
            paraphrased_word = get_first_synonym(word, pos=wordnet.ADJ)
        elif pos_tag.startswith('RB'):
            paraphrased_word = get_first_synonym(word, pos=wordnet.ADV)
        else:
            paraphrased_word = word  # Keep the word unchanged if it's not one of the main POS categories

        paraphrased_text.append(paraphrased_word)

    # Join the paraphrased words into a single string
    paraphrased_sentence = ' '.join(paraphrased_text)

    # Capitalize the first letter of each sentence
    sentences = nltk.sent_tokenize(paraphrased_sentence)  # Split into sentences
    capitalized_sentences = [s.capitalize() for s in sentences]

    # Join the capitalized sentences back together
    final_paraphrase = ' '.join(capitalized_sentences)

    return final_paraphrase

@app.post("/generate/")
async def paraphrase_text(request: ParaphraseRequest):
    try:
        paraphrased = paraphrase(request.text)
        return {"original": request.text, "generated_text": paraphrased}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

# To run the server, use the command: uvicorn your_filename:app --reload
