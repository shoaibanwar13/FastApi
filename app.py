from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, constr, validator
import random
import re
import nltk
from nltk.corpus import wordnet
from textblob import TextBlob

nltk.download('wordnet')  # Download WordNet for synonyms

app = FastAPI()

class UnpredictableTextGenerator:
    def __init__(self, text):
        self.chain = {}
        self.words = self.tokenize(text)
        self.add_to_chain()

    def tokenize(self, text):
        return re.findall(r'\b\w+\b|[.,!?]', text)

    def add_to_chain(self):
        for i in range(len(self.words) - 2):
            current_word = self.words[i]
            next_word = self.words[i + 1]
            following_word = self.words[i + 2]
            if (current_word, next_word) in self.chain:
                if following_word not in self.chain[(current_word, next_word)]:
                    self.chain[(current_word, next_word)].append(following_word)
            else:
                self.chain[(current_word, next_word)] = [following_word]

    def generate_text(self, input_length):
        required_length = max(1, int(input_length * 1.2))  # 120% of input length
        
        start_pair = random.choice(list(self.chain.keys()))
        sentence = [start_pair[0], start_pair[1]]

        while len(sentence) < required_length:
            current_pair = (sentence[-2], sentence[-1])
            next_words = self.chain.get(current_pair, None)
            if not next_words:
                break
            next_word = random.choice(next_words)
            sentence.append(next_word)

        # Capitalize the first word
        if sentence:
            sentence[0] = sentence[0].capitalize()

        # Join to form a sentence
        output_sentence = ' '.join(sentence)

        # Ensure the output is at least the required length
        while len(output_sentence.split()) < required_length:
            next_word = random.choice(self.words)  # Use a random word if needed
            sentence.append(next_word)
            output_sentence = ' '.join(sentence)

        # Clean up punctuation
        output_sentence = re.sub(r'\s+[.,!?]', lambda match: match.group(0).strip(), output_sentence)
        output_sentence = re.sub(r'([.!?]){2,}', r'\1', output_sentence)
        
        # Ensure sentence ends with a period, exclamation, or question mark
        if output_sentence and output_sentence[-1] not in ['.', '!', '?']:
            output_sentence += '.'

        # Replace some words with synonyms to enhance the text's variability
        output_sentence = self.replace_with_synonyms(output_sentence)

        # Correct grammar using TextBlob
        output_sentence = self.correct_grammar(output_sentence)

        return output_sentence

    def replace_with_synonyms(self, sentence):
        words = sentence.split()
        new_words = []
        for word in words:
            # Skip common words from synonym replacement
            if word.lower() in ['the', 'and', 'of', 'to', 'in', 'a', 'an']:
                new_words.append(word)
                continue

            synonyms = wordnet.synsets(word)
            if synonyms:
                synonym = synonyms[0].lemmas()[0].name()
                if synonym != word:
                    new_words.append(synonym.replace('_', ' '))
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        return ' '.join(new_words)

    def correct_grammar(self, sentence):
        # Use TextBlob for grammar correction
        blob = TextBlob(sentence)
        return str(blob.correct())

    def format_output(self, original_text, generated_text):
        output_lines = []
        for line in original_text.strip().split('\n'):
            if line.startswith('#'):
                output_lines.append(line)
            else:
                output_lines.append(generated_text)
        return '\n'.join(output_lines)

# Define input data model
class TextRequest(BaseModel):
    language: str
    text: constr(max_length=5000)
    length: int = 100

    @validator('text')
    def check_minimum_length(cls, v):
        if len(v.split()) < 10:
            raise ValueError("Text must be at least 10 words long.")
        if not v[0].isupper():
            raise ValueError("Text must start with an uppercase letter.")
        return v

@app.post("/generate/")
async def generate_text(request: TextRequest):
    input_length = len(request.text.split())
    
    generator = UnpredictableTextGenerator(request.text)
    
    generated_text = generator.generate_text(input_length)
    formatted_output = generator.format_output(request.text, generated_text)

    return {"language": request.language, "generated_text": formatted_output}

# Run the application
# Use the command below to run the server:
# uvicorn your_filename:app --reload 
