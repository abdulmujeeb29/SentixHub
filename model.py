import math
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from nltk.tokenize import WhitespaceTokenizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# # # Download NLTK resources
# # # nltk.download('punkt')
# # # nltk.download('stopwords')
# # # nltk.download('wordnet')

# # # Load the data
#data = pd.read_csv('IMDB Dataset.csv')

# # # Define a function to clean and preprocess the text
def preprocess_text(text):
    # Remove HTML tags and URLs
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text).lower()
    
    # Tokenize the text into words
    words = word_tokenize(text)
    
    # Remove stopwords and lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    
    # Join the words back into a string
    return ' '.join(words)

# # # Apply the preprocessing function to the 'review' column
#data['review'] = data['review'].apply(preprocess_text)


# #  #encoding labels into 1's and 0's , i.e  numerical representations.
# reviews = data['review'].values
# labels = data['sentiment'].values
# encoder = LabelEncoder()
# encoded_labels = encoder.fit_transform(labels)

# # #splitting and stratification 
# # # used test size for the proportion to be 80% train set and 20% test set 
# train_sentences, test_sentences, train_labels, test_labels = train_test_split(
#     reviews, encoded_labels, stratify=encoded_labels, test_size=0.2)


# # ## Manual implementation of naive byers classifier 
# # Create a CountVectorizer
# vec = CountVectorizer(max_features = 3000)
# X = vec.fit_transform(train_sentences)
# vocab = vec.get_feature_names_out()
# X = X.toarray()

# # Calculate word counts
# word_counts = {}
# for l in range(2):
#     word_counts[l] = defaultdict(lambda: 0)
# for i in range(X.shape[0]):
#     l = train_labels[i]
#     for j in range(len(vocab)):
#         word_counts[l][vocab[j]] += X[i][j]


def laplace_smoothing(n_label_items, vocab, word_counts, word, text_label):
    a = word_counts[text_label][word] + 1
    b = n_label_items[text_label] + len(vocab)
    return math.log(a/b)


def group_by_label(x, y, labels):
    data = {}
    for l in labels:
        data[l] = x[np.where(y == l)]
    return data
def fit(x, y, labels):
    n_label_items = {}
    log_label_priors = {}
    n = len(x)
    grouped_data = group_by_label(x, y, labels)
    for l, data in grouped_data.items():
        n_label_items[l] = len(data)
        log_label_priors[l] = math.log(n_label_items[l] / n)
    return n_label_items, log_label_priors



w_tokenizer = WhitespaceTokenizer()
def predict(n_label_items, vocab, word_counts, log_label_priors, labels, x):
    result = []
    for text in x:
        label_scores = {l: log_label_priors[l] for l in labels}
        words = set(w_tokenizer.tokenize(text))
        for word in words:
            if word not in vocab: continue
            for l in labels:
                log_w_given_l = laplace_smoothing(n_label_items, vocab, word_counts, word, l)
                label_scores[l] += log_w_given_l
        result.append(max(label_scores, key=label_scores.get))
    return result


# labels = [0,1]
# n_label_items, log_label_priors = fit(train_sentences,train_labels,labels)
# pred = predict(n_label_items, vocab, word_counts, log_label_priors, labels, test_sentences)
# print("Accuracy of prediction on test set : ", accuracy_score(test_labels,pred))


# # # Create a Multinomial Naive Bayes classifier
# clf = MultinomialNB()
# clf.fit(X, train_labels)


# # # Save the trained model
# joblib.dump(clf, 'trained_model.pkl')
# joblib.dump(vec, 'vectorizer.pkl') 