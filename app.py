from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
import jieba  # For Chinese text segmentation
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

class ParaphraseRequest(BaseModel):
    language: str
    text: str
    target_length: int = 100

# Paraphrasing for non-Chinese text
def get_first_synonym(word, pos=None):
    """Retrieve the first synonym from WordNet if available and appropriate."""
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

    return final_paraphrase

# Chinese text generation using jieba and Markov chains
class ChineseTextGenerator:
    def _init_(self, text):
        self.chain = {}
        self.words = self.tokenize(text)
        self.add_to_chain()
