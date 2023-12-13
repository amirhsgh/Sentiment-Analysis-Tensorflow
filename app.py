import os
from nltk.corpus import stopwords
from nltk import word_tokenize
from string import punctuation
import random
import time
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Embedding, Conv1D, MaxPool1D, Dropout
from keras.layers import concatenate
import pickle
import streamlit as st

model = load_model('textcnn.h5')
with open('tokenizer.h5', 'rb') as f:
    tokenizer = pickle.load(f)
stop_words = stopwords.words('english')


st.title("Sentiment analysis")
text = st.text_input("Enter a text for prediction")
if text:
    st.toast(":green[your text send successfully to model]")
    with st.spinner('Waiting for model...'):
        time.sleep(2)
    st.toast(':green[Done!]') 
    tokens = word_tokenize(text)
    translator = str.maketrans('', '', punctuation)
    tokens = [w.translate(translator) for w in tokens]
    tokens = [w for w in tokens if not w in stop_words]
    text = ' '.join(tokens)
    text = tokenizer.texts_to_sequences([text])[0]
    text = pad_sequences([text], maxlen=1824, padding='post')
    pred = model.predict(text)
    if float(pred) > 0.5:
        st.subheader(':green[Positive] with this predication {:0.2f}'.format(pred[0][0]))
    else:
        st.subheader(':red[Negative] with this predication {:0.2f}'.format(1 - pred[0][0]))
else:
    st.toast(":red[please send a text]")