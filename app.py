from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
import jieba  # For Chinese text segmentation
import nltk
from nltk.corpus import wordnet
from fastapi.middleware.cors import CORSMiddleware
import logging
import re  # For punctuation spacing adjustments
from textblob import TextBlob  # For additional text correction

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
do_not_replace = {"is", "are", "has", "have", "was", "were", "be", "been", "am", "does", "did", "had"}

def get_synonyms(word, pos=None, max_synonyms=5):
    """
    Retrieve a list of synonyms for a given word and part of speech.
    """
    synonyms = wordnet.synsets(word, pos=pos)
    valid_synonyms = set()
    for syn in synonyms:
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            # Filter out invalid synonyms (numbers, long words, same as original)
            if (
                synonym.lower() != word.lower()
                and not any(char.isdigit() for char in synonym)
                and len(synonym) < 20
            ):
                valid_synonyms.add(synonym)
            if len(valid_synonyms) >= max_synonyms:
                break
        if len(valid_synonyms) >= max_synonyms:
            break
    return list(valid_synonyms)

def get_contextual_synonym(word, pos=None):
    """
    Get a contextually suitable synonym for a word, avoiding unnatural choices.
    """
    synonyms = get_synonyms(word, pos)
    if synonyms:
        num_synonyms = min(5, len(synonyms))
        weights = [0.5, 0.2, 0.15, 0.1, 0.05][:num_synonyms]
        return random.choices(synonyms[:num_synonyms], weights=weights, k=1)[0]
    return word

def expand_text_with_filler(sentences):
    """
    Expand sentences by adding relevant filler or elaboration to increase length.
    """
    filler_phrases = [
        "Interestingly,",
        "It is worth mentioning that",
        "Moreover,",
        "In addition to that,",
        "As a matter of fact,",
        "Notably,"
    ]
    expanded_sentences = []
    for sentence in sentences:
        if random.random() > 0.5:
            filler = random.choice(filler_phrases)
            sentence = f"{filler} {sentence}"
        expanded_sentences.append(sentence)
    return expanded_sentences

def paraphrase(text: str) -> str:
    words = nltk.word_tokenize(text)
    paraphrased_text = []

    for word in words:
        if word.lower() in do_not_replace:
            paraphrased_word = word
        else:
            pos_tag = nltk.pos_tag([word])[0][1]
            if pos_tag.startswith('NN'):
                paraphrased_word = get_contextual_synonym(word, pos=wordnet.NOUN)
            elif pos_tag.startswith('VB'):
                paraphrased_word = get_contextual_synonym(word, pos=wordnet.VERB)
            elif pos_tag.startswith('JJ'):
                paraphrased_word = get_contextual_synonym(word, pos=wordnet.ADJ)
            elif pos_tag.startswith('RB'):
                paraphrased_word = get_contextual_synonym(word, pos=wordnet.ADV)
            else:
                paraphrased_word = word

        paraphrased_text.append(paraphrased_word)

    # Join the words back into a string
    paraphrased_sentence = ' '.join(paraphrased_text)

    # Correct spacing for punctuation
    paraphrased_sentence = re.sub(r'\s+([,.!?])', r'\1', paraphrased_sentence)
    paraphrased_sentence = re.sub(r'([.!?])([^\s])', r'\1 \2', paraphrased_sentence)
    paraphrased_sentence = re.sub(r"\b(\w+)\s+'(\w+)\b", r"\1'\2", paraphrased_sentence)

    # Split into sentences and capitalize each one
    sentences = nltk.sent_tokenize(paraphrased_sentence)
    expanded_sentences = expand_text_with_filler(sentences)
    capitalized_sentences = [s.capitalize() for s in expanded_sentences]

    # Use TextBlob for grammar correction
    blob = TextBlob(' '.join(capitalized_sentences))
    corrected_sentences = str(blob.correct())

    # Join the sentences back into a single text, ensuring it is 110% of the input length
    final_paraphrase = ' '.join(corrected_sentences.split())
    while len(final_paraphrase.split()) < len(text.split()) * 1.1:
        final_paraphrase += " " + random.choice(expanded_sentences)
    
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
            output = f"{heading}\n{paraphrased}" if heading else paraphrased
            return {"language": request.language, "original": request.text, "generated_text": output}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
