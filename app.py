from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
import jieba  # For Chinese text segmentation
import nltk
from nltk.corpus import wordnet
from fastapi.middleware.cors import CORSMiddleware
import logging
import re
import requests

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

# Hugging Face API details for BART model
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": "Bearer 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"}

# Function to query Hugging Face API
def query_bart(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        logger.error(f"Hugging Face API error: {response.text}")
        return None

# Define a request model for the input, including heading and text
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

    # Correct spacing for punctuation
    paraphrased_sentence = re.sub(r'\s+([,.!?])', r'\1', paraphrased_sentence)
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

# Function to detect heading and separate it from paragraph
def detect_heading_and_paragraph(text: str):
    lines = text.strip().split('\n')
    heading = None
    if len(lines) > 0:
        first_line = lines[0].strip()
        if len(first_line.split()) <= 6 and first_line[-1] not in ".!?":
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
            input_length = len(list(jieba.cut(paragraph)))
            generated_text = generator.generate_text(input_length)
            output = f"{heading}\n{generated_text}" if heading else generated_text
            return {"language": request.language, "generated_text": output}
        else:
            logger.info(f"Paraphrasing text for language: {request.language}")
            paraphrased = paraphrase(paragraph)

            # Use the BART model for further refinement
            logger.info(f"Sending text to BART for further processing.")
            bart_response = query_bart({"inputs": paraphrased})

            if bart_response and "summary_text" in bart_response:
                final_output = bart_response["summary_text"]
            else:
                final_output = paraphrased

            output = f"{heading}\n{final_output}" if heading else final_output
            return {"language": request.language, "original": request.text, "generated_text": output}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
