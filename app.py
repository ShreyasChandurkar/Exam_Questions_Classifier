import streamlit as st
import PyPDF2
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect, LangDetectException
import contractions
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('punkt')
# Load the model and tokenizer from Hugging Face Hub
model_id = "ChetanIngle/Bert-Bloom-toxonomy"
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# Define text cleaning functions
def strip_all_entities(text):
    text = re.sub(r'\r|\n', ' ', text.lower())
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    text = re.sub(r'[^\x00-\x7f]', '', text)
    banned_list = string.punctuation
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def clean_hashtags(tweet):
    new_tweet = re.sub(r'(\s+#[\w-]+)+\s*$', '', tweet).strip()
    new_tweet = re.sub(r'#([\w-]+)', r'\1', new_tweet).strip()
    return new_tweet

def filter_chars(text):
    return ' '.join('' if ('$' in word) or ('&' in word) else word for word in text.split())

def remove_mult_spaces(text):
    return re.sub(r"\s\s+", " ", text)

def filter_non_english(text):
    try:
        lang = detect(text)
    except LangDetectException:
        lang = "unknown"
    return text if lang == "en" else ""

def expand_contractions(text):
    return contractions.fix(text)

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_short_words(text, min_len=2):
    words = text.split()
    long_words = [word for word in words if len(word) >= min_len]
    return ' '.join(long_words)

def replace_elongated_words(text):
    regex_pattern = r'\b(\w+)((\w)\3{2,})(\w*)\b'
    return re.sub(regex_pattern, r'\1\3\4', text)

def remove_repeated_punctuation(text):
    return re.sub(r'[\?\.\!]+(?=[\?\.\!])', '', text)

def remove_extra_whitespace(text):
    return ' '.join(text.split())

def remove_spaces_tweets(tweet):
    return tweet.strip()

def remove_short_tweets(tweet, min_words=3):
    words = tweet.split()
    return tweet if len(words) >= min_words else ""

def clean_tweet(tweet):
    tweet = expand_contractions(tweet)
    tweet = filter_non_english(tweet)
    tweet = strip_all_entities(tweet)
    tweet = clean_hashtags(tweet)
    tweet = filter_chars(tweet)
    tweet = remove_mult_spaces(tweet)
    tweet = remove_numbers(tweet)
    tweet = remove_short_words(tweet)
    tweet = replace_elongated_words(tweet)
    tweet = remove_repeated_punctuation(tweet)
    tweet = remove_extra_whitespace(tweet)
    tweet = remove_spaces_tweets(tweet)
    tweet = remove_short_tweets(tweet)
    tweet = ' '.join(tweet.split())
    return tweet



def make_prediction(question, model, tokenizer):
    question = clean_tweet(question)
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        predictions = torch.argmax(logits, dim=-1)
        label_id = predictions.item()
    return label_id

# Streamlit App
st.title("Bloom's Toxonomy Level Detector")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    pdfReader = PyPDF2.PdfReader(uploaded_file)
    numPages = len(pdfReader.pages)
    text = ""
    for i in range(numPages):
        page = pdfReader.pages[i]
        text += page.extract_text()

    sample = ''
    for line in text.split('\n'):
        sample += " " + line

    questions = []
    matches = re.finditer(r"Q\s?(\d)\s+(.+?)(?=\sQ\s?\d|\Z)", sample, flags=re.S)
    for match in matches:
        question_num, question_body = match.groups()
        sub_questions = []
        sub_matches = re.finditer(r"(\w\))\s+(.+?)(?=\s\w\)|\Z)", question_body, flags=re.S)
        for sub_match in sub_matches:
            sub_questions.append(sub_match.group(2))
        questions.append((question_num, sub_questions))

    corpus = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
    predictions = []
    for num, sub_questions in questions:
        sub_predictions = []
        for sub_question in sub_questions:
            predicted_category = make_prediction(sub_question, model, tokenizer)
            sub_predictions.append(corpus[predicted_category] + " - " + clean_tweet(sub_question))
        predictions.append((num, sub_predictions ))

    for num, subs in predictions:
        st.write(f"Q{num} Predictions:")
        for sub in subs:
            st.write(f"  - {sub}")
