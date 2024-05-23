# for basic operations
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import io
import matplotlib.pyplot as plt
import seaborn as sns
import json

# for data prepocessing
import string
import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# for providing path
import os

# Enable logging for gensim - optional
import logging
import warnings

#for modelling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from collections import Counter
from wordcloud import WordCloud

#evaluation
from gensim.models import CoherenceModel

#Extended file
from function import describe_detail, plot_histogram, plot_kde, remove_tweet_special, remove_number, remove_punctuation, remove_whitespace_LT, remove_whitespace_multiple, remove_single_char, word_tokenize_wrapper, freqDist_wrapper, stopwords_removal, normalized_term, remove_punct, stemmed_wrapper, get_stemmed_term, elbow_method, silhouette_method, k_means, get_top_features_cluster, plotWords, euclidean_distance, calculate_centroid, calculate_s_within, calculate_s_between,  calculate_dbi

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_excel("dataset.xlsx")
    df = df.drop('No', axis=1)
    return df
df = load_data()

def load_data2():
    df2 = pd.read_csv('output_stemmed.csv')
    return df2
df2 = load_data2()

def load_data3():
    df3 = pd.read_csv('output_stemmed_with_8_labels.csv')
    return df3
df3 = load_data3()

def main(df):
    # Main Page Design
    st.title(':mailbox_with_mail: :blue[TESIS]')
    st.header('_:blue[Text Classification Model]_')
    st.sidebar.title("Menu")
    menu = st.sidebar.selectbox("Exploratory Data Analysis :", ["*****-----*****-----*****-----*****", 
                                                          "Statistic Descriptive", 
                                                          "Check Data Distributions",
                                                          "Data Prepocessing"])
    
    menu2 = st.sidebar.selectbox("Modeling:", ["*****-----*****-----*****-----*****", 
                                                     "LED K-Means", 
                                                     "LDA Model",
                                                     "Model Evaluation"])
    # Menu Functions
    st.set_option('deprecation.showPyplotGlobalUse', False)
    if menu == "- - - - -" and menu2 == "- - - - -" :
       st.write('''TESIS is a data science project ...''')
    if menu == "Statistic Descriptive" :
       describe_detail(df)   
    if menu == "Check Data Distributions" :
       # Sidebar menu to select plot type
       plot_type = st.radio("Select Plot Type", ("Histogram", "KDE Plot"))
       if plot_type == "Histogram":
          st.sidebar.subheader("Histogram Settings")
          column = st.sidebar.selectbox("Select Column", df.columns)
          color = st.sidebar.color_picker("Select Color")
          xlabel = st.sidebar.text_input("X-axis Label", "Length")
          ylabel = st.sidebar.text_input("Y-axis Label", "Frequency")
          title = st.sidebar.text_input("Plot Title", "Histogram")
          plot_histogram(df, column, xlabel, ylabel, title, color)

       elif plot_type == "KDE Plot":
            st.sidebar.subheader("KDE Plot Settings")
            column1 = st.sidebar.selectbox("Select Column 1", df.columns)
            column2 = st.sidebar.selectbox("Select Column 2", df.columns)
            color1 = st.sidebar.color_picker("Select Color for Column 1")
            color2 = st.sidebar.color_picker("Select Color for Column 2")
            xlabel = st.sidebar.text_input("X-axis Label", "Length")
            ylabel = st.sidebar.text_input("Y-axis Label", "Density")
            title = st.sidebar.text_input("Plot Title", "KDE Plot")
            plot_kde(df, column1, column2, xlabel, ylabel, title, color1, color2)
    if menu == "Data Prepocessing" :
       # Load stopwords
       list_stopwords = set(stopwords.words('indonesian'))
       # Load additional stopwords
       list_stopwords.update(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 'kalo', 'amp', 'biar', 'bikin', 'bilang','gak', 'ga', 'krn', 'nya', 'nih', 'sih', 'si', 'tau', 'tdk', 'tuh', 'utk', 'ya','jd', 'jgn', 'sdh', 'aja', 'n', 't', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt','&amp', 'yah', 'wkwk', 'ini', 'invensi', 'sehingga'])
        # Load additional stopwords from txt file
       with open("stopword.txt", "r") as f:
           additional_stopwords = f.read().splitlines()
       list_stopwords.update(additional_stopwords)
        
       # Load normalization dictionary
       normalized_word = pd.read_excel("normalisasi-V1.xlsx")
       normalized_word_dict = dict(zip(normalized_word['slang'], normalized_word['formal']))

       st.subheader("Preprocessed Text")
       st.write("Case Folding :")
       text = df['text'].str.lower()
       st.write(text)
       st.write("Tokenizing :")
       text = text.apply(remove_tweet_special)
       # Remove special characters, URLs, mentions, and hashtags
       text = text.apply(remove_tweet_special)  
       # Remove numbers
       text = text.apply(remove_number)
       # Remove punctuation
       text = text.apply(remove_punctuation)
       # Remove leading and trailing whitespaces
       text = text.apply(remove_whitespace_LT)
       # Remove multiple whitespaces
       text = text.apply(remove_whitespace_multiple)
       # Remove single characters
       text = text.apply(remove_single_char)
       # Tokenizing
       text = text.apply(word_tokenize_wrapper)
       st.write(text)
       st.write("Frequency Tokens :")
       # Apply the frequency distribution function
       df['tokens_fdist'] = df['text'].apply(lambda x: freqDist_wrapper(x.split()))
       # Convert FreqDist to a JSON string
       df['tokens_fdist'] = df['tokens_fdist'].apply(lambda fd: json.dumps(fd.most_common()))
       st.write(df[['text', 'tokens_fdist']])
       st.write("Filtering (Stopword Removal) :")
       text = text.apply(lambda x: stopwords_removal(x, list_stopwords))
       st.write(text)
       st.write("Text Normalization :")
       text = text.apply(lambda x: normalized_term(x, normalized_word_dict))
       st.write(text)
       st.write("Remove puntuation for saved file")
       text = text.apply(remove_punct)
       # Delete comma
       text = text.apply(lambda x: [word for word in x if word != ','])
       st.write(text)
       # Save preprocessed text to CSV
       text.apply(lambda x: ' '.join(x)).to_csv('Streamlit_processed_text.csv', index=False, sep=' ')
       st.success("File saved successfully: Streamlit_processed_text.csv")
       term_dict = {}
       for document in text:
           for term in document:
               if term not in term_dict:
                  term_dict[term] = ' '
       st.write("---------------------------------------")
       st.write("The length of term dictionary : ",len(term_dict))
       st.write("---------------------------------------")
       text = text.apply(lambda x: get_stemmed_term(x))
       st.write("Stemmed Text :")
       st.write(text)
       text.apply(lambda x: ' '.join(x)).to_csv('Streamlit_output_stemmed.csv', index=False, sep=' ')
       st.success("File saved successfully: Streamlit_output_stemmed.csv")
       
    if menu2 == "LED K-Means" :
       st.subheader("TF-IDF")
       # Text vectorization using TF-IDF
       vectorizer = TfidfVectorizer()
       X = vectorizer.fit_transform(df2['stemmed'])
       tf_idf = pd.DataFrame(data = X.toarray(),columns=vectorizer.get_feature_names_out())
       final_df = tf_idf
       st.write("{} rows".format(final_df.shape[0]))
       st.write(final_df.T.nlargest(432, 0))
       
       st.subheader("Elbow Method")
       elbow_method(X)
       
       st.subheader("Silhouette Method")
       silhouette_method(X)
       
       st.subheader("LED K-Means")
       k = 8
       # Apply the  LED K-Means algorithm
       centroids, labels = k_means(X.toarray(), k)
       # Add the algorithm result labels into a new column in the dataframe
       df2['cluster_label'] = labels
       st.write(df2)
       
       st.subheader("Most Common Words in Each Cluster")
       # Get the top features in each cluster
       dfs = get_top_features_cluster(X.toarray(), labels, vectorizer, n_feats=10)
       # Plot the top words in each cluster
       plotWords(dfs, n_feats=10)
       
       st.subheader("Cluster Distribution")
       # Count the frequency of each cluster label and sort by index
       cluster_label_counts = df2['cluster_label'].value_counts().sort_index()
       # Plot the distribution of cluster labels with 'Set2' color palette
       plt.figure(figsize=(8, 6))
       sns.barplot(x=cluster_label_counts.index, y=cluster_label_counts.values, palette='Set2')
       plt.title('Distribution of Cluster Labels')
       plt.xlabel('Cluster Label')
       plt.ylabel('Frequency')
       plt.xticks(rotation=45)
       plt.grid(axis='y')
       st.pyplot()
       # Save the updated dataframe into a new CSV file
       df2['cluster_label'].to_csv('Streamlit_output_stemmed_with_8_labels.csv', index=False)
       st.success("File saved successfully: Streamlit_output_stemmed_with_8_labels.csv")
       
    if menu2 == "LDA Model" :
       st.subheader("Cluster Label")
       # Iterate through unique cluster labels
       for label in df3['cluster_label'].unique():
           cluster = df3[df3['cluster_label'] == label][['stemmed']]
           st.write('Cluster {} :'.format(label))
           st.write(cluster.head())
           st.write('\n')
       # Preprocessing function
       def preprocess_text(text):
           tokens = word_tokenize(text)
           tokens = [word.lower() for word in tokens if word.isalpha()]  # Remove non-alphabetic tokens
           tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
           tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
           return tokens
       # Initialize variables
       stop_words = set(stopwords.words('indonesian'))
       lemmatizer = WordNetLemmatizer()
       topics_per_cluster = {}
       
       st.subheader("LDA Model")
       # Iterate through clusters
       for label in df3['cluster_label'].unique():
           cluster_data = df3[df3['cluster_label'] == label]['stemmed']
           processed_data = [preprocess_text(doc) for doc in cluster_data]
           dictionary = corpora.Dictionary(processed_data)
           corpus = [dictionary.doc2bow(doc) for doc in processed_data]
           lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=1, passes=10)
           topics = lda_model.print_topics(num_words=10)
           topics_per_cluster[label] = topics
           st.write(f"Cluster {label} Topics:")
           for topic in topics:
               st.write(topic)
           st.write("\n")
       
       st.subheader("LDA Visualization")
       st.write("Cluster 0")
       with open("cluster_0_lda_vis.html", "r") as f:
                   html_content0 = f.read()
       components.html(html_content0, height=600, scrolling=True)
       st.write("Cluster 1")
       with open("cluster_1_lda_vis.html", "r") as f:
                   html_content1 = f.read()
       components.html(html_content1, height=600, scrolling=True)
       st.write("Cluster 2")
       with open("cluster_2_lda_vis.html", "r") as f:
                   html_content2 = f.read()
       components.html(html_content2, height=600, scrolling=True)
       st.write("Cluster 3")
       with open("cluster_3_lda_vis.html", "r") as f:
                   html_content3 = f.read()
       components.html(html_content3, height=600, scrolling=True)
       st.write("Cluster 4")
       with open("cluster_4_lda_vis.html", "r") as f:
                   html_content4 = f.read()
       components.html(html_content4, height=600, scrolling=True)
       st.write("Cluster 5")
       with open("cluster_5_lda_vis.html", "r") as f:
                   html_content5 = f.read()
       components.html(html_content5, height=600, scrolling=True)
       st.write("Cluster 6")
       with open("cluster_6_lda_vis.html", "r") as f:
                   html_content6 = f.read()
       components.html(html_content6, height=600, scrolling=True)
       st.write("Cluster 7")
       with open("cluster_7_lda_vis.html", "r") as f:
                   html_content7 = f.read()
       components.html(html_content7, height=600, scrolling=True)
       
       st.subheader("Top 10 Words in Cluster")
       # Function to generate and display bar chart of top 10 words for each topic
       def generate_bar_chart(top_words, topic_num):
           plt.figure(figsize=(10, 5))
           plt.barh(range(len(top_words)), [val[1] for val in top_words], align='center', color='lightgreen')
           plt.yticks(range(len(top_words)), [val[0] for val in top_words])
           plt.gca().invert_yaxis()
           plt.xlabel('Word Frequency')
           plt.ylabel('Words')
           plt.title(f'Top 10 Words in Cluster {label}')
           st.pyplot()
       # Initialize variables
       top_words_per_cluster = {}
       # Iterate through clusters
       for label in df3['cluster_label'].unique():
           cluster_data = df3[df3['cluster_label'] == label]['stemmed']
           processed_data = [preprocess_text(doc) for doc in cluster_data]
           dictionary = corpora.Dictionary(processed_data)
           corpus = [dictionary.doc2bow(doc) for doc in processed_data]
           lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=1, passes=10)
           # Get top 10 words for each topic in the cluster
           top_words_per_topic = {}
           for topic_num in range(lda_model.num_topics):
               top_words = lda_model.show_topic(topic_num, topn=10)
               top_words_per_topic[topic_num] = top_words
               generate_bar_chart(top_words, topic_num)
           top_words_per_cluster[label] = top_words_per_topic
       
       st.subheader("Wordcloud")
       # Function to generate and display word cloud
       def generate_wordcloud(words, title):
           wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(words))
           plt.figure(figsize=(7, 3))
           plt.imshow(wordcloud, interpolation='bilinear')
           plt.axis('off')
           plt.title(title)
           st.pyplot()
              # Initialize variables
       top_words_per_cluster = {}
       # Iterate through clusters
       for label in df3['cluster_label'].unique():
           cluster_data = df3[df3['cluster_label'] == label]['stemmed']
           processed_data = [preprocess_text(doc) for doc in cluster_data]
           dictionary = corpora.Dictionary(processed_data)
           corpus = [dictionary.doc2bow(doc) for doc in processed_data]
           lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=1, passes=10)
           # Generate word cloud for the cluster
           cluster_words = [word for doc in processed_data for word in doc]
           word_counts = Counter(cluster_words)
           generate_wordcloud(word_counts, f'Word Cloud for Cluster {label}')

    if menu2 == "Model Evaluation" :
       st.subheader("LED K-means Evaluation")
       vectorizer = TfidfVectorizer()
       X = vectorizer.fit_transform(df2['stemmed'])
       tf_idf = pd.DataFrame(data=X.toarray(), columns=vectorizer.get_feature_names_out())
       final_df = tf_idf
       k = 8
       centroids, labels = k_means(X.toarray(), k)
       df2['cluster_label'] = labels
       st.write(df2)
       # Convert DataFrame to a format accepted by calculate_dbi function
       clusters = [final_df[df2['cluster_label'] == i].values for i in range(k)]
       # Calculate DBI
       dbi_score = calculate_dbi(final_df.values, clusters)
       st.write("DBI Score:", dbi_score)
       st.subheader("DBI Score Graph")
       # Try different values of k
       k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
       dbi_scores = []
       for k in k_values:
           # Apply K-Means algorithm
           centroids, labels = k_means(X.toarray(), k)
           # Add the algorithm result labels into a new column in the dataframe
           df['cluster_label'] = labels
           # Convert DataFrame to a format accepted by calculate_dbi function
           clusters = [final_df[df['cluster_label'] == i].values for i in range(k)]
           # Calculate DBI
           dbi_score = calculate_dbi(final_df.values, clusters)
           dbi_scores.append(dbi_score)
           st.write("DBI Score for k = ", k, ":", dbi_score)
       # Plot DBI scores for different k values
       plt.figure(figsize=(10, 6))
       plt.plot(k_values, dbi_scores, marker='o', linestyle='-')
       plt.title('DBI Scores for Different k Values')
       plt.xlabel('Number of Clusters (k)')
       plt.ylabel('DBI Score')
       plt.xticks(k_values)
       plt.grid(True)
       st.pyplot()
       
       st.subheader("LDA Model Evaluation")
       # Preprocessing function
       def preprocess_text(text):
           tokens = word_tokenize(text)
           tokens = [word.lower() for word in tokens if word.isalpha()]  # Remove non-alphabetic tokens
           tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
           tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
           return tokens
       # Initialize variables
       stop_words = set(stopwords.words('indonesian'))
       lemmatizer = WordNetLemmatizer()
       topics_per_cluster = {}       
       # Initialize variables
       perplexity_per_cluster = {}
       coherence_per_cluster = {}
       # Iterate through clusters
       for label in df3['cluster_label'].unique():
           cluster_data = df3[df3['cluster_label'] == label]['stemmed']
           processed_data = [preprocess_text(doc) for doc in cluster_data]
           dictionary = corpora.Dictionary(processed_data)
           corpus = [dictionary.doc2bow(doc) for doc in processed_data]
           lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=2, passes=10)
           st.subheader(f"Cluster {label} Evaluation")
           # Compute perplexity
           perplexity = lda_model.log_perplexity(corpus)
           perplexity_per_cluster[label] = perplexity
           st.write(f"Cluster {label} Perplexity: {perplexity}")
           # Compute coherence score
           coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_data, dictionary=dictionary, coherence='u_mass')
           coherence = coherence_model_lda.get_coherence()
           coherence_per_cluster[label] = coherence
           st.write(f"Cluster {label} Coherence Score: {coherence}")
           st.write("\n")
       
       
if __name__=="__main__":
    df = load_data()
    main(df)
