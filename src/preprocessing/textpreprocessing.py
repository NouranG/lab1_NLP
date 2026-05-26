#importing necessary libraries
import re
import html
import string
import token
import unicodedata

import nltk
from nltk import text
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize

import pandas as pd
import nltk
import os



stop_words = set(stopwords.words("english"))

#keeping a set of negation words to handle negations in text
negation_words = {
    "not",
    "no",
    "nor",
    "don't",
    "didn't",
    "won't",
    "isn't",
    "wasn't",
    "aren't",
    "couldn't",
    "shouldn't",
    "wouldn't",
    "hasn't",
    "haven't"
}

stop_words = stop_words - negation_words

# stemmer and lemmatizer initialization
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def normalize_unicode(text):
    """
    Normalize unicode characters.
    """

    return unicodedata.normalize("NFKC", text)


def lowercase_text(text):
    """
    Convert text to lowercase.
    """

    return text.lower()


def remove_html(text):
    """
    Remove HTML tags.
    """

    text = html.unescape(text)
    text = re.sub(r"<.*?>", " ", text)

    return text


def remove_urls(text):
    """
    Remove URLs.
    """

    return re.sub(r"http\S+|www\S+", " ", text)


def remove_mentions(text):
    """
    remove twitter mentions.
    """

    return re.sub(r"@", "", text)


def process_hashtags(text):
    """
    Remove #
    """

    return re.sub(r"#", "", text)

def remove_emojis(text):
    """
    Remove emojis.
    """

    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )

    return emoji_pattern.sub(r" ", text)


def remove_numbers(text):
    """
    Remove numbers.
    """

    return re.sub(r"\d+", " ", text)

def remove_punctuation(text):
    """
    Remove punctuation.
    """

    translator = str.maketrans("", "", string.punctuation)

    return text.translate(translator)


def remove_extra_whitespace(text):
    """
    Remove repeated spaces.
    """
    return re.sub(r"\s+", " ", text).strip()


#removing stop words and applying stemming and lemmatization
def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens
#removing stop words
def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]
#applying stemming and lemmatization
def stem_tokens(tokens):
    return [stemmer.stem(word) for word in tokens]
def lemmatize_tokens(tokens):
    return " ".join([lemmatizer.lemmatize(t) for t in tokens])

#main preprocessing function that applies all the above steps
def preprocess_text(
    text,remove_stopword_flag=True,use_stemming=False,use_lemmatization=True
):
    """
    Complete preprocessing pipeline.
    """


    # Handling nulls
  

    if pd.isna(text):
        return ""

    # Convert to string
    text = str(text)

    #text normalization steps:

    text = normalize_unicode(text)
    text = lowercase_text(text)
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_mentions(text)
    text = process_hashtags(text)
    text = remove_emojis(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = remove_extra_whitespace(text)

#tokenization

    tokens = tokenize_text(text)

# Removing stop words, applying stemming and lemmatization
    if remove_stopword_flag:
        tokens = remove_stopwords(tokens)
    if use_stemming:
        tokens = stem_tokens(tokens)
    if use_lemmatization:
        tokens = lemmatize_tokens(tokens)

    cleaned_text = "".join(tokens)

    return cleaned_text

#full dataframe preprocessing function

def preprocess_dataframe(
    dataframe,
    text_column,
    output_column="clean_text",
    remove_stopword_flag=True,
    use_stemming=False,
    use_lemmatization=True
):
    """
    Apply preprocessing to entire dataframe column.
    """

    dataframe[output_column] = dataframe[text_column].apply(
        lambda text: preprocess_text(
            text=text,
            remove_stopword_flag=remove_stopword_flag,
            use_stemming=use_stemming,
            use_lemmatization=use_lemmatization
        )
    )

    return dataframe




