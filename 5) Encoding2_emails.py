import pandas as pd 
import numpy as np

# import csv dataset
email = pd.read_csv("Emails.csv")
print(email.head())

# Check the number null values the dataset
print(email.isnull().sum().sum())

emails = email['Message']

'''----------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------'''

# Remove the punctuation in the emails
import string 
emails = emails.str.translate(
    str.maketrans("", "", string.punctuation)
)


'''----------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------'''

# Remove the stopwords in the csv files 
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("English"))

# Split the sentence into words 
split_emails = emails.str.split()

# Remove stopwords
clean_emails = []

for email in split_emails:
    cleaned_words = []

    for word in email:
        if word.lower() not in stop_words:
            cleaned_words.append(word)

    clean_emails.append(cleaned_words)

'''----------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------'''

'''
SnowballStemmer
    use on the sentence to convert words in their root form
    Snowball Stemmer: Snowball Stemmer is generally an improved version of Porter Stemmer.
'''
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("english")

stemmed_emails = []

for email in clean_emails:
    stemmed_words = []
    for word in email:
        stemmed_word = stemmer.stem(word)
        stemmed_words.append(stemmed_word)

    stemmed_emails.append(stemmed_words)

print(stemmed_emails)


'''----------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------'''

'''
Remove the numbers from the list 
'''
import re
clean_stemmed_emails = []

for email in stemmed_emails:
    cleaned_words = []

    for word in email:
        word = re.sub(r'\d+', '', word)

        if word:
            cleaned_words.append(word)

    clean_stemmed_emails.append(cleaned_words)

print(clean_stemmed_emails)

'''----------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------------'''

df = pd.DataFrame({
    'Message': clean_stemmed_emails
})
print(df.head())








'''
5. CountVectorizer in NLP
    CountVectorizer is a technique from scikit-learn used to convert text data into numerical data so that Machine Learning algorithms can process it.
    It is also called the Bag of Words (BoW) approach.

    1. Why do we need CountVectorizer?
        ML models cannot directly understand:
            "I love Python"
            "Python is easy"
        CountVectorizer converts these sentences into numbers based on how many times each word appears.
'''

#Count Vectorization is a technique used in Natural Language Processing (NLP) to convert text data into numbers by counting how many times each word appears.
df['Message'] = df['Message'].apply(
    lambda x: ' '.join(x) if isinstance(x, list) else str(x)
)

print(df.head())

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Message'])
print(X.toarray())
print(X.shape)