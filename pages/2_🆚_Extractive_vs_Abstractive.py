import streamlit as st
import pandas as pd
import numpy as np
import nltk
#Download for first time
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import regex as re
from nltk.stem.snowball import SnowballStemmer, PorterStemmer
from datasets import load_dataset
import copy
from rouge import Rouge
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from heapq import nlargest
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

st.cache(show_spinner=False)
def load_model():
    #model_name = 'google/pegasus-large'
    model_name = 'google/pegasus-billsum'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #run using local model
    tokenizer = PegasusTokenizer.from_pretrained(model_name,use_auth_token=True)
    model = PegasusForConditionalGeneration.from_pretrained(model_name, max_position_embeddings=2000,use_auth_token=True).to(torch_device)
    #tokenizer = PegasusTokenizer.from_pretrained("local_pegasus-billsum_tokenizer", use_auth_token=True)
    #model = PegasusForConditionalGeneration.from_pretrained("local_pegasus-billsum_tokenizer_model", max_position_embeddings=2000, use_auth_token=True).to(torch_device)
    #tokenizer = PegasusTokenizer.from_pretrained("local_pegasus-billsum_tokenizer")
    #model = PegasusForConditionalGeneration.from_pretrained("local_pegasus-billsum_tokenizer_model", max_position_embeddings=2000).to(torch_device)
    return model,tokenizer

model,tokenizer = load_model()

#run this the first time and use the local model for faster runtime
#tokenizer.save_pretrained("local_pegasus-billsum_tokenizer")
#model.save_pretrained("local_pegasus-billsum_tokenizer_model")


en_stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")

def preprocessing(string):
    '''
    Given 1 single str, 
    returns a cleaned sentence
    '''
    # take out symbols
    string = re.sub(r'\([^)]*\)', '', string)
    string = re.sub('\n', '', string)
    string = re.sub('<n>', '', string)
    string = re.sub(' +', ' ', string)
    string = re.sub(r'[^\w\s\.\,]', '', string)
    string = re.sub('\.(?!\s|\d|$)', '. ', string)
    string = string.lower()
    return string

def delete_leading_white_spaces(string):
    return re.sub(r'^[ \t]+', '', string)

def clear_leading_white_tab(string):
    '''
    Give 1 single string, clean out all the tabs (4 white spaces)
    '''
    if len(string) == 0 : return ""
    if string[:4] == '    ':
        return clear_leading_white_tab(string[4:])
    else:
        return string[:4] + clear_leading_white_tab(string[4:])

def further_split(ugly_string):
    '''
    Given a string with newline \n in them,
    Returns a list of actual sentences
    '''
    lines = ugly_string.split('\n')
    cleaned = []
    for line in lines:
        cleaned.append(clear_leading_white_tab(line))
    condensed = []
    for i in range(len(cleaned)):
        p = cleaned[i][0] == '(' and cleaned[i][2] == ')'
        if p or cleaned[i][:3] == '``(':
            condensed.append(cleaned[i])
        elif len(condensed) == 0:
            condensed.append(cleaned[i])
        else:
            condensed[-1] += cleaned[i]
    return condensed

def split_right(long_string):
    '''
    Given a long string (a whole bill),
    Performs sentence tokenization (rather than tokenizing based on period)
    '''
    result = []
    paragraphs = long_string.split('\n\n')
    for paragraph in paragraphs:
        if '\n' in paragraph:
            split_ps = further_split(paragraph)
            for sent in split_ps:
                result.append(sent)
        else:
            result.append(paragraph)
    return result


def stemming(list_of_tokenized_strings):
    '''
    Given a tokenized sentences as a list, 
    returns a 2d list of stemmed sentences
    '''
    processed_sentences = []
    for i in range(len(list_of_tokenized_strings)):
        words = word_tokenize(list_of_tokenized_strings[i])
        stemmed_words = []
        for j in range(len(words)):
            word = stemmer.stem(words[j])
            if word not in en_stopwords:
                stemmed_words.append(word)
        processed_sentences.append(stemmed_words) 
    return processed_sentences

def create_freq_matrix(preprocessed_sentences, stemmed_sentences):
    '''
    Given two 2d arrays preprocessed_sentences and stemmed_sentences,
    returns a nested fequency matrix in the form of 
    {'sent' : {'word1': freq1, 'word2': freq2}}
    '''
    freq_matrix = {}
    for i in range(len(stemmed_sentences)):
        freq_table = {}
        for j in range(len(stemmed_sentences[i])):
            word = stemmed_sentences[i][j]
            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1
        sent = preprocessed_sentences[i]
        freq_matrix[sent] = freq_table
    return freq_matrix

def tf(freq_matrix):
    # value is the frequency dictionary
    tf_matrix = copy.deepcopy(freq_matrix)
    for sent, freq_dict in tf_matrix.items():
        for key, value in freq_dict.items():
            freq_dict[key] = value/len(freq_dict)
    return tf_matrix

def num_sent_per_word(stemmed_sentences):
    '''
    Given a 2d arrays stemmed_sentences, return a dict with 
    '''
    num_sent_per_word = {}
    for i in range(len(stemmed_sentences)):
        for j in range(len(stemmed_sentences[i])):
            word = stemmed_sentences[i][j]
            if word in num_sent_per_word:
                num_sent_per_word[word] += 1
            else:
                num_sent_per_word[word] = 1
    return num_sent_per_word

def idf(freq_matrix, num_sent_per_word, num_sent):
    idf = copy.deepcopy(freq_matrix)
    for sent, freq_dict in idf.items():
        for key, value in freq_dict.items():
            freq_dict[key] = np.log(num_sent / num_sent_per_word[key])
    return idf

def tf_idf(tf, idf):
    tf_idf = {}
    for (k,v), (k2,v2) in zip(tf.items(), idf.items()):
        tf_idf_table = {}
        for (key, tf_v), (key2, idf_v) in zip(v.items(), v2.items()):
            tf_idf_table[key] = tf_v * idf_v
        tf_idf[k] = tf_idf_table
    return tf_idf

def score_sentences(tf_idf_matrix):
    sent_scores = {}
    
    for sent, tf_idf in tf_idf_matrix.items():
        sent_score = 0
        sent_len = len(tf_idf)
        for word, tf_idf_score in tf_idf.items():
            sent_score += tf_idf_score
        sent_scores[sent] = sent_score / sent_len
    return sent_scores

def average_sent_score(sentences_score):
    total = 0
    for sent, sent_score in sentences_score.items():
        total += sent_score
    avg = total/len(sentences_score)
    return avg

def generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence in sentenceValue and sentenceValue[sentence] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary

def everything_generate_summary(original_string, multiplier):
    '''
    Given a string of a bill and a multiplier for generating the summary,
    returns a summary
    '''
    # tokenize 
    example_sentences = split_right(original_string)
    # preprocess
    cleaned_sentences = []
    for i in range(len(example_sentences)):
        cleaned_sentences.append(preprocessing(example_sentences[i]))
    for i in range(len(cleaned_sentences)):
        cleaned_sentences[i] = delete_leading_white_spaces(cleaned_sentences[i])
    # stem
    stemmed_sentences = stemming(example_sentences)
    # calculate tf-idf
    freq_matrix = create_freq_matrix(example_sentences, stemmed_sentences)
    tf_matrix = tf(freq_matrix)
    nums_sent_per_word = num_sent_per_word(stemmed_sentences)
    idf_matrix = idf(freq_matrix, nums_sent_per_word, len(stemmed_sentences))
    tf_idf_matrix = tf_idf(tf_matrix, idf_matrix)
    # setting a metric for generating summary 
    sentences_score = score_sentences(tf_idf_matrix)
    threshold = average_sent_score(sentences_score)
    summary = generate_summary(example_sentences, sentences_score, multiplier * threshold)
    return summary

def get_rouge_scores(final_summary, original_text):
    rouge = Rouge()
    scores = rouge.get_scores(final_summary, original_text)
    df = pd.DataFrame.from_dict(scores[0])
    return df

def sklearn_generate_summary(original_string, n):
    # tokenize 
    example_sentences = split_right(original_string)
    # preprocess
    cleaned_sentences = []
    for i in range(len(example_sentences)):
        cleaned_sentences.append(preprocessing(example_sentences[i]))
    for i in range(len(cleaned_sentences)):
        cleaned_sentences[i] = delete_leading_white_spaces(cleaned_sentences[i])
    # vectorize
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)
    # score
    scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    summary_sentences = nlargest(n, range(len(scores)), key=scores.__getitem__)
    result_vector = []
    for i in sorted(summary_sentences):
        result_vector.append(example_sentences[i])
    result = " ".join(result_vector)
    
    return result


# The actual app
# dataset = load_dataset("billsum", split = "test")
# dataset = pd.DataFrame(dataset)
dataset = pd.read_csv("test_sample.csv")
txt = dataset.iat[0, 0]
original_summary = dataset.iat[0, 1]

st.set_page_config(page_title="Text Summarizations Side by Side", layout="wide")
st.markdown("# Text Summarizations Side by Side")

if st.button('Randomly generate a Bill Example'):
    my_num = random.randrange(len(dataset))
    txt = dataset.iat[my_num, 0]
    original_summary = dataset.iat[my_num, 1]
else:
    pass

column1, column2 = st.columns(2)
with column1:
    txt = st.text_area('Text', txt, height = 250)
with column2:
    original_summary = st.text_area('Corresponding summary', original_summary, height = 250)



# txt = st.text_area('Text', txt, height = 250)
# original_summary = st.text_area('Corresponding summary', original_summary, height = 250)


col1, col2, col3 = st.columns(3)
with col1:
    st.header("TF-IDF from scratch:")
    my_multiplier = st.slider('Please input a multiplier value:', 1.0, 1.5)
    first_summary = everything_generate_summary(txt, my_multiplier)
    st.write("#### Summary:")
    st.write(first_summary)
    st.write("#### Rouge score:", get_rouge_scores(first_summary, txt))

with col2:
    st.header("TF-IDF from Sklearn:")
    num_of_sentences = st.number_input('How many sentences do you want to generate?', 1)
    second_summary = sklearn_generate_summary(txt, num_of_sentences)
    st.write("#### Summary:")
    st.write(second_summary)
    st.write("#### Rouge score:", get_rouge_scores(second_summary, txt))

with col3:
    st.header("Abstractive summary:")
    min_l = st.slider('Please input a a minimum length (words) for the summary:', 1, 50, step=1,value=20)
    if(st.button("Generate")):
        txt_pre = preprocessing(txt)
        txt_cleaned = delete_leading_white_spaces(txt_pre)
        batch = tokenizer.prepare_seq2seq_batch(txt_cleaned, truncation=True, padding='longest',return_tensors='pt')
        translated = model.generate(**batch,min_length=min_l, max_new_tokens = 100)
        abs_summary = tokenizer.batch_decode(translated, skip_special_tokens=True)
        st.write("#### Summary:")
        st.write (abs_summary[0], height = 400, placeholder="Abstractive Summary", unsafe_allow_html=True)
        st.write("#### Rouge score:", get_rouge_scores(abs_summary[0], txt))
    else:
        pass
