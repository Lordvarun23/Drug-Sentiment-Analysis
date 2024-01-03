import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from PIL import Image
import joblib

StopWords = stopwords.words("english")
ps = PorterStemmer()
wnl = WordNetLemmatizer()

import streamlit as st

# Home Page
def home():
    st.title("Drug Sentiment Analyzer")
    st.write("Drug Sentiment Analysis aims to analyze public sentiment towards various drugs based on user-generated content such as social media posts, reviews, or forum discussions. The goal is to understand the general sentiment (positive, negative) associated with specific drugs and gain insights into public opinions.")

# Visualization Page
def visualization():
    st.title("Visualization Page")

    image = Image.open('images//Frequency_distribution.png')
    st.image(image, caption="Frequency distribution of labels", use_column_width=True)

    image = Image.open('images//medical_conditions.png')
    st.image(image, caption="Frequency distribution of Medical Conditions", use_column_width=True)

    image = Image.open('images//positive_drug_wordcloud.png')
    st.image(image, caption="positive sentiment of drug wordcloud", use_column_width=True)

    st.subheader("Inferences:")
    st.write('''
    1. We have 25,485 records with no null values.
2. Top 5 drugs that are examined are 'Bupropion', 'Lamotrigine', 'Liraglutide','Levothyroxine', 'Sodium hyaluronate'.
3. Depression is the prevalant medical condition followed by Bi polar disorder.
4. With word cloud level analysis of top 5 drug and corresponding reviews. It also suggests that Bupropion is used for Depression, Liraglutide is used for diabetes etc.
5. Sentiment distribution is not balanced, with label 0 = 14,859 and label 1 = 10,986.
6. Analysing the Negative sentiments of few condition drugs it suggests some side affects of using those drugs for this condition.
7. Analysing the positive sentiments of few condition drugs it suggests no side affects of using those drugs for this condition.
    ''')
# Prediction Page
def prediction():
    st.title("Prediction Page")
    uploaded_file = st.file_uploader("Choose a csv file")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()

        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

        # To read file as string:
        string_data = stringio.read()

        # Can be used wherever a "file-like" object is accepted:
        df = pd.read_csv(uploaded_file)
        st.write("Input Data")
        st.write(df)

        def Tokenisation(text, sep=" "):
            '''
            Function to tokenise the given text and lowercasing.
            ---------------------------------------
            :text : word sequence
            :sep : by default space character
            ---------------------------------------
            :return : list of words
            '''
            text = text.lower()
            text = re.sub('[^A-Za-z0-9]+', '', text)
            tokens = text.split(sep)

            return tokens

        def RemoveStopWords(tokens):
            '''
            Function to remove stop words from the given text tokens.
            ---------------------------------------
            :tokens : list of words after tokenisation
            ---------------------------------------
            :return : list of words after removing stop words
            '''

            res = []
            for token in tokens:
                if token not in StopWords:
                    res.append(token)
                else:
                    continue

            return res

        def Stemming(tokens):
            '''
            Function to perform stemmign from the given text tokens.
            ---------------------------------------
            :tokens : list of words after tokenisation
            ---------------------------------------
            :return : list of words after stemming
            '''

            res = []

            for token in tokens:
                res.append(wnl.lemmatize(token))

            return res

        def BagOfWords(tokens, unique_tokens):
            '''
            Function to get bag of words from the given array of array tokens.
            ---------------------------------------
            :tokens : matrix of words after tokenisation
            :unique_tokens : array of unique words
            ---------------------------------------
            :return : vectorized format
            '''
            matrix = []
            for record in tokens:
                array = [0] * len(unique_tokens)
                for word in record:
                    if word not in unique_tokens:
                        continue
                    else:
                        array[unique_tokens.index(word)] = 1

                matrix.append(array)
            matrix = np.array(matrix)

            print("Shape of BOW Matrix:", matrix.shape)

            return matrix

        # tokenisation
        tokens = [Tokenisation(i) for i in list(df['review'].values)]

        # Remove Stop words
        removed_tokens = [RemoveStopWords(i) for i in tokens]

        # lemmetization
        stemmed_tokens = [Stemming(i) for i in removed_tokens]

        unique = np.load("unique.npy")
        BoW = BagOfWords(stemmed_tokens, list(unique))

        one_hot = list(np.load("onehot.npy"))
        onehot = []
        for i in range(len(df)):
            temp = [0] * len(one_hot)

            temp[one_hot.index("drugName_" + df.iloc[i]['drugName'])] = 1

            temp[one_hot.index("condition_" + df.iloc[i]['condition'])] = 1

            onehot.append(temp)

        final = np.hstack((np.array(BoW), np.array(onehot)))

        clf = joblib.load('logreg_drug_sentiment (1).pkl')
        st.write("Predicted Labels:")
        st.write(clf.predict(final))


# App
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", options=["Home", "Visualization", "Prediction"])

    if page == "Home":
        home()
    elif page == "Visualization":
        visualization()
    elif page == "Prediction":
        prediction()

if __name__ == "__main__":
    main()
