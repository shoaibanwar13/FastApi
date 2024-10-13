from fastapi import FastAPI
from pydantic import BaseModel
import nltk
from nltk.corpus import wordnet
import random
import re

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')

app = FastAPI()

class TextInput(BaseModel):
    language: str
    text: str

def extract_named_entities(text):
    """Extract named entities from the text."""
    entities = set()
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    named_entities = nltk.ne_chunk(pos_tags)
    
    for chunk in named_entities:
        if hasattr(chunk, 'label'):
            entities.add(' '.join(c[0] for c in chunk.leaves()))
    
    return entities

def get_synonyms(word, pos=None):
    """Retrieve synonyms for a given word."""
    synonyms = wordnet.synsets(word, pos=pos)
    word_synonyms = set()
    
    for syn in synonyms:
        for lemma in syn.lemmas():
            if lemma.name() != word and len(lemma.name()) < 20 and lemma.name().isalpha():
                word_synonyms.add(lemma.name().replace('_', ' '))
    
    return list(word_synonyms)

def replace_with_synonyms(word, pos_tag, named_entities):
    """Replace a word with its synonym if applicable."""
    if word in named_entities or word[0].isupper() or not word.isalpha():
        return word

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
    if synonyms:
        return random.choice(synonyms)
    return word

def humanize_sentence(sentence, named_entities):
    """Humanize a single sentence by replacing words with synonyms."""
    words = nltk.word_tokenize(sentence)
    tagged_words = nltk.pos_tag(words)
    paraphrased_words = []

    for word, tag in tagged_words:
        paraphrased_words.append(replace_with_synonyms(word, tag, named_entities))

    paraphrased_sentence = ' '.join(paraphrased_words).replace('  ', ' ').strip()
    return paraphrased_sentence.capitalize() + '.'

def humanize_paragraph(text: str) -> str:
    """Humanize a paragraph by processing each sentence."""
    named_entities = extract_named_entities(text)
    sentences = nltk.sent_tokenize(text)
    paraphrased_sentences = [humanize_sentence(sentence, named_entities) for sentence in sentences]
    return ' '.join(paraphrased_sentences)

@app.post("/humanize/")
async def humanize_text(input: TextInput):
    """Endpoint to humanize input text."""
    humanized_text = humanize_paragraph(input.text)
    return {
        "language": input.language,
        "original": input.text,
        "generated_text": humanized_text
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
