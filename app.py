from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
import jieba  # For Chinese text segmentation
import nltk
from nltk.corpus import wordnet
from fastapi.middleware.cors import CORSMiddleware
import logging
import re  # For punctuation spacing adjustments

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
    language: str
    text: str
    target_length: int = 100

# Paraphrasing for non-Chinese text
do_not_replace = {"is", "are", "has", "have", "was", "were", "be", "been", "am", "does", "did", "had"}

def get_first_synonym(word, pos=None):
    synonyms = wordnet.synsets(word, pos=pos)
    if synonyms:
        lemma = synonyms[0].lemmas()[0].name()
        if not any(char.isdigit() for char in lemma) and len(lemma) < 20:
            return lemma.replace('_', ' ')
    return word

def paraphrase(text: str) -> str:
    month_names = {
        "january", "february", "march", "april", "may", "june", 
        "july", "august", "september", "october", "november", "december"
    }

    words = nltk.word_tokenize(text)
    paraphrased_text = []

    for word in words:
        # Preserve capitalization and certain words
        if word.isupper() or word.lower() in month_names or word.lower() in do_not_replace:
            paraphrased_word = word
        else:
            pos_tag = nltk.pos_tag([word])[0][1]
            if pos_tag.startswith('NNP'):
                paraphrased_word = word.capitalize()
            elif pos_tag.startswith('NN'):
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

    # Join the words back into a string
    paraphrased_sentence = ' '.join(paraphrased_text)

    # Correct spacing for punctuation (e.g., no space before commas, periods, etc.)
    paraphrased_sentence = re.sub(r'\s+([,.!?])', r'\1', paraphrased_sentence)

    # Correct contractions and possessives
    paraphrased_sentence = re.sub(r"\b(\w+)\s+'(\w+)\b", r"\1'\2", paraphrased_sentence)

    # Split into sentences and capitalize each one properly
    sentences = nltk.sent_tokenize(paraphrased_sentence)
    capitalized_sentences = [s.capitalize() for s in sentences]

    # Handle capitalization for specific words after initial processing
    final_paraphrase = []
    for sentence in capitalized_sentences:
        words_in_sentence = sentence.split()
        capitalized_words = [
            word.capitalize() if word.lower() in month_names or pos_tag.startswith('NNP') else word
            for word, pos_tag in nltk.pos_tag(words_in_sentence)
        ]
        final_paraphrase.append(' '.join(capitalized_words))

    # Join the sentences back into a single string
    final_paraphrase_text = ' '.join(final_paraphrase)

    # Further cleanup for grammar corrections
    final_paraphrase_text = re.sub(r'\bi\s+', 'I ', final_paraphrase_text)
    final_paraphrase_text = re.sub(r'(\s+)([,.!?])', r'\2', final_paraphrase_text)
    final_paraphrase_text = re.sub(r'\s+(\'s)', r"'s", final_paraphrase_text)

    return final_paraphrase_text

# API endpoint for text generation/paraphrasing
@app.post("/generate/")
async def generate_text(request: ParaphraseRequest):
    try:
        heading, paragraph = detect_heading_and_paragraph(request.text)

        if request.language.lower() == "chinese":
            logger.info(f"Generating text for Chinese input: {paragraph}")
            generator = ChineseTextGenerator(paragraph)
            input_length = len(list(jieba.cut(paragraph)))
            generated_text = generator.generate_text(input_length)
            output = f"{heading}\n{generated_text}" if heading else generated_text
            return {"language": request.language, "generated_text": output}
        else:
            logger.info(f"Paraphrasing text for language: {request.language}")
            paraphrased = paraphrase(paragraph)
            output = f"{heading}\n{paraphrased}" if heading else paraphrased
            return {"language": request.language, "original": request.text, "generated_text": output}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

