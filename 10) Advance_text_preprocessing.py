'''
Advanced Text Preprocessing in NLP and Deep Learning
    Advanced text preprocessing means converting raw, messy text into a form that NLP or deep-learning models can understand effectively.
    A modern NLP pipeline can look like:
        Raw Text → Cleaning → Normalization → Tokenization → Subword Tokenization → Vocabulary/IDs → Padding/Attention Mask → 
        Embeddings → Deep Learning Model
    Unlike traditional ML, deep-learning NLP does not require aggressive preprocessing. Models such as BERT and GPT use tokenization and embeddings to learn many linguistic patterns themselves.

    
1. Basic vs Advanced Text Preprocessing
    Basic preprocessing
        "I LOVE Machine Learning!!! 😊"
                ↓
        "i love machine learning"
                ↓
        ["i", "love", "machine", "learning"]    
    
    Typical operations:
        Lowercasing
        Removing punctuation
        Removing URLs
        Removing HTML
        Removing extra spaces
        Stopword removal
        Stemming
        Lemmatization
        Tokenization
    
    Advanced preprocessing
'''

# Unicode Normalization: Text can contain visually identical characters represented differently internally.
import unicodedata
text = "café"
normalized = unicodedata.normalize("NFKC", text)          #For many NLP pipelines, NFKC is useful for reducing compatibility variations.

print(normalized)

# Handling Contractions                   don't  → do not,  can't  → cannot
import contractions
text = "I don't like this movie."
expanded = contractions.fix(text)

print(expanded)

# Handling URLs: Raw URLs can introduce thousands of unnecessary variations.
import re
text = "Visit https://www.niti.gov.in/search/node?keys=internship for more information."
text = re.sub(r"https?://\S+|www\.\S+", "", text)
print(text)

# Repeated Characters: Social-media text frequently contains: sooooo good, goooood, heyyyyy,  nooooo
text = "This is soooo good"
text = re.sub(r'(.)\1{2,}', r'\1\1', text)
print(text)

# POS-Aware Preprocessing: POS means Part of Speech.                  Named Entity Recognition
#  noun, verb,adjective, adverb, pronoun
# Load the English language model
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The boy is running quickly.")

for token in doc:
    print(token.text, token.pos_)

# < ------------------------------------------------------------------------------------------------------------------------------ >

# Complete Advanced Preprocessing Example: Here is a practical preprocessing function:
import re
import unicodedata
import emoji


def advanced_preprocess(text):

    # 1. Convert to string
    text = str(text)

    # 2. Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # 3. Remove HTML
    text = re.sub(r"<[^>]+>", " ", text)

    # 4. Replace URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URL ",
        text
    )

    # 5. Replace emails
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " EMAIL ",
        text
    )

    # 6. Replace mentions
    text = re.sub(
        r"@\w+",
        " USER ",
        text
    )

    # 7. Convert emojis to text
    text = emoji.demojize(text)

    # 8. Normalize repeated characters
    text = re.sub(
        r"(.)\1{2,}",
        r"\1\1",
        text
    )

    # 9. Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # 10. Remove leading/trailing spaces
    text = text.strip()

    return text


text = """
OMG!!! Visit https://example.com 😍😍
Contact me at test@gmail.com
This product is soooo good!!!
"""

result = advanced_preprocess(text)

print(result)
