from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
import jieba  # For Chinese text segmentation
import nltk
from nltk.corpus import wordnet
from fastapi.middleware.cors import CORSMiddleware
import logging
import re  # For punctuation spacing adjustments
from spellchecker import SpellChecker  # Importing the spell checker

# Download necessary NLTK data files
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware for cross-origin resource sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a request model for input, including language and text
class ParaphraseRequest(BaseModel):
    language: str
    text: str
    target_length: int = 100

# Set of words to avoid replacing
do_not_replace = {"is", "are", "has", "have", "was", "were", "be", "been", "am", "does", "did", "had"}

def get_first_synonym(word, pos=None):
    """Retrieve the first synonym of a word based on its part-of-speech."""
    synonyms = wordnet.synsets(word, pos=pos)
    if synonyms:
        lemma = synonyms[0].lemmas()[0].name()
        if not any(char.isdigit() for char in lemma) and len(lemma) < 20:
            return lemma.replace('_', ' ')
    return word

def correct_verb_tense(tagged_word, word):
    """Adjust verb tense based on POS tags."""
    pos_tag = tagged_word[1]
    if pos_tag.startswith('VBD') or pos_tag.startswith('VBN'):
        return word if word.endswith('ed') else f"{word}ed"
    elif pos_tag.startswith('VBZ'):
        return word if word.endswith('s') else f"{word}s"
    return word

def paraphrase(text: str) -> str:
    """Paraphrase English text with synonym replacement and spell correction."""
    spell = SpellChecker()
    words = nltk.word_tokenize(text)
    tagged_words = nltk.pos_tag(words)
    paraphrased_text = []

    for tagged_word in tagged_words:
        word, pos_tag = tagged_word
        corrected_word = spell.correction(word) or word

        if corrected_word.lower() in do_not_replace:
            paraphrased_word = corrected_word
        else:
            # Get synonyms or adjust verbs based on POS tags
            if pos_tag.startswith('NN'):
                paraphrased_word = get_first_synonym(corrected_word, pos=wordnet.NOUN)
            elif pos_tag.startswith('VB'):
                paraphrased_word = correct_verb_tense(tagged_word, get_first_synonym(corrected_word, pos=wordnet.VERB))
            elif pos_tag.startswith('JJ'):
                paraphrased_word = get_first_synonym(corrected_word, pos=wordnet.ADJ)
            elif pos_tag.startswith('RB'):
                paraphrased_word = get_first_synonym(corrected_word, pos=wordnet.ADV)
            else:
                paraphrased_word = corrected_word

        paraphrased_text.append(paraphrased_word)

    paraphrased_sentence = ' '.join(paraphrased_text)
    paraphrased_sentence = re.sub(r'\s+([,.!?])', r'\1', paraphrased_sentence)
    paraphrased_sentence = re.sub(r'([.!?])([^\s])', r'\1 \2', paraphrased_sentence)
    paraphrased_sentence = re.sub(r"\b(\w+)\s+'(\w+)\b", r"\1'\2", paraphrased_sentence)

    sentences = nltk.sent_tokenize(paraphrased_sentence)
    capitalized_sentences = [s.capitalize() for s in sentences]
    final_paraphrase = ' '.join(capitalized_sentences)

    return final_paraphrase

# Chinese text generation using jieba and Markov chains
class ChineseTextGenerator:
    def __init__(self, text):
        self.chain = {}
        self.words = self.tokenize(text)
        self.add_to_chain()

    def tokenize(self, text):
        return list(jieba.cut(text))

    def add_to_chain(self):
        for i in range(len(self.words) - 2):
            current_pair = (self.words[i], self.words[i + 1])
            next_word = self.words[i + 2]
            self.chain.setdefault(current_pair, []).append(next_word)

    def generate_text(self, input_length):
        if input_length < 10:
            return "输入的文本不足以生成新的内容。请提供更长的文本。"

        required_length = max(1, int(input_length * 1.2))
        if not self.chain:
            return "未能生成内容，链为空。"

        start_pair = random.choice(list(self.chain.keys()))
        sentence = [start_pair[0], start_pair[1]]

        while len(sentence) < required_length:
            current_pair = (sentence[-2], sentence[-1])
            next_words = self.chain.get(current_pair)
            if not next_words:
                break
            next_word = random.choice(next_words)
            sentence.append(next_word)

        return ''.join(sentence)

# Function to detect heading and separate it from paragraph
def detect_heading_and_paragraph(text: str):
    lines = text.strip().split('\n')
    heading = None
    if lines:
        first_line = lines[0].strip()
        if len(first_line.split()) <= 6 and not first_line[-1] in ".!?":
            heading = first_line
            paragraph = '\n'.join(lines[1:]).strip()
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

        if request.language.lower() == "chinese":
            logger.info(f"Generating text for Chinese input: {paragraph}")
            generator = ChineseTextGenerator(paragraph)
            generated_text = generator.generate_text(request.target_length)
        else:
            logger.info(f"Generating paraphrase for English input: {paragraph}")
            generated_text = paraphrase(paragraph)

        return {"heading": heading, "generated_text": generated_text}
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
