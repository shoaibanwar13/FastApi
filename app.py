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
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

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
def get_synonyms(word, pos=None):
    # Fetch synonyms from wordnet
    synonyms = wordnet.synsets(word, pos=pos)
    word_synonyms = [lemma.name().replace('_', ' ') for syn in synonyms for lemma in syn.lemmas()]
    word_synonyms = [syn for syn in set(word_synonyms) if syn != word and len(syn) < 20 and syn.isalpha()]
    
    return word_synonyms

def replace_with_synonyms(word, pos_tag):
    # Determine the part of speech tag and fetch synonyms accordingly
    pos = None
    if pos_tag.startswith('NN'):
        pos = wordnet.NOUN
    elif pos_tag.startswith('VB'):
        pos = wordnet.VERB
    elif pos_tag.startswith('JJ'):
        pos = wordnet.ADJ
    elif pos_tag.startswith('RB'):
        pos = wordnet.ADV

    synonyms = get_synonyms(word, pos=pos)
    # Choose a synonym if available, otherwise keep the original word
    if synonyms:
        return random.choice(synonyms)
    return word

def humanize_sentence(sentence):
    words = nltk.word_tokenize(sentence)
    tagged_words = nltk.pos_tag(words)
    paraphrased_words = []

    for word, tag in tagged_words:
        # Replace nouns, verbs, adjectives, and adverbs with synonyms to humanize the text
        if tag.startswith(('NN', 'VB', 'JJ', 'RB')):
            paraphrased_words.append(replace_with_synonyms(word, tag))
        else:
            paraphrased_words.append(word)

    # Shuffle parts of the sentence slightly to create variation while keeping meaning
    if len(paraphrased_words) > 5:
        random.shuffle(paraphrased_words[:3])  # Shuffle the first few words for variety

    paraphrased_sentence = ' '.join(paraphrased_words)
    paraphrased_sentence = paraphrased_sentence.capitalize()
    
    return paraphrased_sentence

def humanize_paragraph(text: str) -> str:
    sentences = nltk.sent_tokenize(text)
    paraphrased_sentences = [humanize_sentence(sentence) for sentence in sentences]
    return ' '.join(paraphrased_sentences)

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
            # Chinese-specific logic
            logger.info(f"Generating text for Chinese input: {paragraph}")
            generator = ChineseTextGenerator(paragraph)
            input_length = len(list(jieba.cut(paragraph)))
            generated_text = generator.generate_text(input_length)
            output = f"{heading}\n{generated_text}" if heading else generated_text
            return {"language": request.language, "generated_text": output}
        else:
            # General paraphrasing logic
            logger.info(f"Paraphrasing text for language: {request.language}")
            paraphrased = humanize_paragraph(paragraph)
            output = f"{heading}\n{paraphrased}" if heading else paraphrased
            return {"language": request.language, "original": request.text, "generated_text": output}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
