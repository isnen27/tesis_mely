# for basic operations
import numpy as np
import pandas as pd
import streamlit as st
import os

#for visualizations
import matplotlib.pyplot as plt
import math
import seaborn as sns
from pandas import plotting
import matplotlib.style as style
style.use("fivethirtyeight")
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as grid_spec
from wordcloud import WordCloud
from collections import Counter
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this

# from pandas_profiling import ProfileReport
from pandas.plotting import parallel_coordinates

# for providing path
import os
#from google.colab import drive
#drive.mount('/content/drive')
#default_dir = "/content/drive/MyDrive/Colab Notebooks/Tesis_Mely"
#os.chdir(default_dir)

#for Feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.datasets import make_classification

# for data prepocessing
import sklearn
from sklearn.model_selection import GridSearchCV, cross_val_score
import string 
import re #regex library
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.stem import SnowballStemmer
import swifter

#for modelling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import LdaModel

#evaluation
from gensim.models import CoherenceModel

#Save model
import joblib
import pickle

# Enable logging for gensim - optional
import logging
import warnings

# Initialize stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Initialize vectorizer
vectorizer = TfidfVectorizer()

def describe_detail(df):
    # Display function with Markdown for titles
    def display_markdown(title):
        st.subheader(f"**{title}**")

    # (a) First five data points
    display_markdown('First five data points')
    st.write(df.head())
    st.write('\n')

    # (b) Random five data points
    display_markdown('Random five data points')
    st.write(df.sample(5))
    st.write('\n')

    # (c) Last five data points
    st.subheader('Last five data points')
    st.write(df.tail())
    st.write('\n')

    # (d) Shape and Size of data set
    shape_size_df = pd.DataFrame({'Shape': [df.shape], 'Size': [df.size]})
    st.subheader('Shape and Size of dataset')
    st.write(shape_size_df)
    print('\n')

    # (e) Data types
    data_types_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
    st.subheader('Data types of columns')
    st.write(data_types_df)
    print('\n')

    # (f) Numerical features in the dataset
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_features:
        st.subheader('Numerical features in the dataset')
        st.write(numerical_features)
        print('\n')

    # (g) Categorical features in the dataset
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_features:
        st.subheader('Categorical features in the dataset')
        st.write(categorical_features)
        print('\n')
    else:
       st.subheader('**No object/category data in dataset.**')
       print('\n')

    # (h) Statistical Description of Columns
    if numerical_features:
        st.subheader('Statistical Description of Numerical Columns')
        st.write(df.describe().T)
        print('\n')

    # (i) Description of Categorical features
    if categorical_features:
        st.subheader('Description of Categorical Features')
        st.write(df.describe(include=['object', 'category']))
        print('\n')

    # (j) Unique class count of Categorical features
    if categorical_features:
        unique_counts_df = pd.DataFrame(df[categorical_features].nunique(), columns=['Unique Count'])
        st.subheader('Unique class count of Categorical features')
        st.write(unique_counts_df)
        print('\n')

    # (k) Missing values in data
    missing_values_df = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
    missing_values_df = missing_values_df[missing_values_df['Missing Values'] > 0]
    if not missing_values_df.empty:
        st.subheader('Missing values in data')
        st.write(missing_values_df)
    else:
        st.subheader('**No missing values found.**')
        st.write(df.isnull().sum())
        print('\n')

    # (l) Unique Value Counts
    unique_values = {}
    for col in df.columns:
        unique_values[col] = df[col].nunique()

    unique_value_counts = pd.DataFrame(unique_values, index=['unique value count']).transpose()

    st.subheader('**Unique Value Counts**')
    st.write(unique_value_counts)

# Function to plot histograms
def plot_histogram(data, column, xlabel, ylabel, title, color):
    plt.figure(figsize=(8, 6))
    sns.histplot(data[column], bins=20, kde=True, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    st.pyplot()

# Function to plot KDE plots
def plot_kde(data, column1, column2, xlabel, ylabel, title, color1, color2):
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data[column1], color=color1)
    sns.kdeplot(data[column2], color=color2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend([column1, column2])
    plt.title(title)
    st.pyplot()

# Function to remove special characters, URLs, mentions, and hashtags
def remove_tweet_special(text):
    text = text.replace('\\t', " ").replace('\\n', " ").replace('\\u', " ").replace('\\', "")
    text = text.encode('ascii', 'replace').decode('ascii')
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
    return text.replace("http://", " ").replace("https://", " ")

# Function to remove numbers
def remove_number(text):
    return re.sub(r"\d+", "", text)

# Function to remove punctuation
def remove_punctuation(text):
    translator = str.maketrans("", "", string.punctuation.replace("-", ""))
    return text.translate(translator)

# Function to remove leading and trailing whitespaces
def remove_whitespace_LT(text):
    return text.strip()

# Function to remove multiple whitespaces into single whitespace
def remove_whitespace_multiple(text):
    return re.sub('\s+', ' ', text)

# Function to remove single characters
def remove_single_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

# Function to tokenize text
def word_tokenize_wrapper(text):
    return word_tokenize(text)

def freqDist_wrapper(text):
    return FreqDist(text)

# Function to remove stopwords
def stopwords_removal(words, list_stopwords):
    return [word for word in words if word not in list_stopwords]

# Function for text normalization
def normalized_term(document, normalized_word_dict):
    return [normalized_word_dict.get(term, term) for term in document]

def remove_punct(document):
    punctuations = '''!()[]{};:'"\,<>./?@#$%^&*_~'''
    return [token for token in document if token not in punctuations]
    
# Function for stemming terms
def stemmed_wrapper(term):
    return stemmer.stem(term)

def get_stemmed_term(document):
    return [stemmed_wrapper(term) for term in document]

# LED K-means function
def k_means(data, k, max_iterations=100):
    # Randomly initialize centroids
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    
    for _ in range(max_iterations):
        # Calculate the distance between each data point and each centroid
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        distances = np.log(distances)
        
        # Group each data point into the group that has the closest centroid
        labels = np.argmin(distances, axis=0)
        
        # Update the centroid position with the average of all data points in the group
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # Check convergence
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, labels

# Function to get the top features in each cluster
def get_top_features_cluster(tf_idf_array, prediction, vectorizer, n_feats):
    labels = np.unique(prediction)
    dfs = []
    for label in labels:
        id_temp = np.where(prediction == label)  # indices for each cluster
        x_means = np.mean(tf_idf_array[id_temp], axis=0)  # returns average score across cluster
        sorted_means = np.argsort(x_means)[::-1][:n_feats]  # indices with top n_feats scores
        best_features = [(vectorizer.get_feature_names_out()[i], x_means[i]) for i in sorted_means]
        df = pd.DataFrame(best_features, columns=['features', 'score'])
        dfs.append(df)
    return dfs

# Function to plot the top words within each cluster
def plotWords(dfs, n_feats, palette='Set2'):
    sns.set_palette(palette)
    for i in range(len(dfs)):
        plt.figure(figsize=(8, 4))
        plt.title(("Most Common Words in Cluster {}".format(i)), fontsize=10, fontweight='bold')
        sns.barplot(x='score', y='features', orient='h', palette='Set2', data=dfs[i][:n_feats])
        st.pyplot()

# DBI function
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

def calculate_centroid(cluster):
    if len(cluster) == 0:
        return None
    return np.mean(cluster, axis=0)

def calculate_s_within(cluster):
    centroid = calculate_centroid(cluster)
    s_within = 0
    for point in cluster:
        s_within += euclidean_distance(point, centroid)
    return s_within / len(cluster)

def calculate_s_between(cluster, other_clusters):
    centroid = calculate_centroid(cluster)
    s_between = 0
    for other_cluster in other_clusters:
        other_centroid = calculate_centroid(other_cluster)
        s_between += euclidean_distance(centroid, other_centroid)
    return s_between / len(other_clusters)

def calculate_dbi(data, clusters):
    k = len(clusters)
    s = np.zeros(k)
    for i in range(k):
        s[i] = calculate_s_within(clusters[i])
    
    dbi = 0
    for i in range(k):
        max_s = -np.inf
        for j in range(k):
            if i != j:
                s_ij = (s[i] + s[j]) / euclidean_distance(calculate_centroid(clusters[i]), calculate_centroid(clusters[j]))
                if s_ij > max_s:
                    max_s = s_ij
        dbi += max_s
    dbi /= k
    return dbi

# Elbow Method
def elbow_method(X, max_clusters=10):
    distortions = []
    for i in range(1, max_clusters + 1):
        centroids, labels = k_means(X.toarray(), i)
        distortions.append(sum(np.min(np.sqrt(((X.toarray() - centroids[:, np.newaxis])**2).sum(axis=2)), axis=0)) / X.shape[0])
    plt.plot(range(1, max_clusters + 1), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')
    st.pyplot()

# Silhouette Method
def silhouette_method(X, max_clusters=10):
    silhouette_scores = []
    for i in range(2, max_clusters + 1):
        centroids, labels = k_means(X.toarray(), i)
        silhouette_avg = silhouette_score(X, labels)
        silhouette_scores.append(silhouette_avg)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method')
    st.pyplot()





