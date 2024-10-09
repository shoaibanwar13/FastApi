from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
import jieba  # For Chinese text segmentation
import nltk
from nltk.corpus import wordnet
from fastapi.middleware.cors import CORSMiddleware
import logging

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

class ParaphraseRequest(BaseModel):
    language: str
    text: str
    target_length: int = 100

# Paraphrasing for non-Chinese text
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

    return final_paraphrase

# Chinese text generation using jieba and Markov chains
class ChineseTextGenerator:
    def __init__(self, text):  # Constructor now accepts 'text'
        self.chain = {}
        self.words = self.tokenize(text)
        self.add_to_chain()

    def tokenize(self, text):
        return list(jieba.cut(text))

    def add_to_chain(self):
        for i in range(len(self.words) - 2):
            current_pair = (self.words[i], self.words[i + 1])
            next_word = self.words[i + 2]
            if current_pair in self.chain:
                if next_word not in self.chain[current_pair]:
                    self.chain[current_pair].append(next_word)
            else:
                self.chain[current_pair] = [next_word]

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

        output_sentence = ''.join(sentence)

        while len(output_sentence) < required_length:
            next_word = random.choice(self.words)
            sentence.append(next_word)
            output_sentence = ''.join(sentence)

        return output_sentence

# API endpoint for text generation/paraphrasing
@app.post("/generate/")
async def generate_text(request: ParaphraseRequest):
    try:
        if request.language.lower() == "chinese":
            # Chinese-specific logic
            logger.info(f"Generating text for Chinese input: {request.text}")
            generator = ChineseTextGenerator(request.text)
            input_length = len(list(jieba.cut(request.text)))
            generated_text = generator.generate_text(input_length)
            return {"language": request.language, "generated_text": generated_text}
        else:
            # General paraphrasing logic
            logger.info(f"Paraphrasing text for language: {request.language}")
            paraphrased = paraphrase(request.text)
            return {"language": request.language, "original": request.text, "generated_text": paraphrased}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
