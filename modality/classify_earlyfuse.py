import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.feature_extraction import text

from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from sklearn import tree

from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import TransformerMixin, BaseEstimator


## load topic model
lda_model = pickle.load(open("lda_5.pkl", 'rb'))


# a simple tokenizer splitting text on whitespace
def whitespace_tokenizer(s):
   return s.split(' ')


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return data_frame[[self.key]]


class Converter(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return data_frame.values.ravel()


class LDATransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return lda_model.transform(data_frame)


df = pd.read_csv("data.csv")
df = pd.get_dummies(df, columns=["INDUSTRY"])

y = df["label"]
X = df.drop(["label"], axis=1)


def get_text(X):
    return X["question"].tolist()

get_surf = FunctionTransformer(lambda x: x[["question_sent_count", "question_tok_count", "question_modality_strong", "question_modality_weak", "question_uncertainty", "question_positive", "question_negative"]], validate=False)
get_fin = FunctionTransformer(lambda x: x[["VOLA_PRIOR", "VIX", "SIZE", "BTM", "SUE", "INDUSTRY_1", "INDUSTRY_2", "INDUSTRY_3", "INDUSTRY_4",  "INDUSTRY_5", "INDUSTRY_6", "INDUSTRY_7", "INDUSTRY_8", "INDUSTRY_9", "INDUSTRY_10", "INDUSTRY_11", "INDUSTRY_12"]], validate=False)

custom_stop_words = text.ENGLISH_STOP_WORDS.union(["'d","'m","'re","'ve","'s","n't"])
vectorizer = TfidfVectorizer(tokenizer=whitespace_tokenizer, stop_words=custom_stop_words, ngram_range=(1,1))
vectorizer_count = CountVectorizer(tokenizer=whitespace_tokenizer, stop_words=custom_stop_words)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)


## early fusion classification pipeline
pipe = Pipeline([
    ('union', FeatureUnion(
        transformer_list=[

        	## surface features
            ('surface', Pipeline([
				('selector', get_surf),
                ('scaler', StandardScaler()),
            ])),

            ## lexical (semantic) features
            ('lexical', Pipeline([
				('selector', ItemSelector(key='question')),
                ('converter', Converter()),
                ('tfidf', vectorizer),
                ('best', TruncatedSVD(n_components=300, random_state=7)),
                ('scaler', StandardScaler()),
            ])),

            ## semantic features
            ('topic', Pipeline([
                ('selector', ItemSelector(key='question')),
                ('converter', Converter()),
                ('count', vectorizer_count),
                ('lda', LDATransformer()),
                ('scaler', StandardScaler()),
            ])),

            ## financial features
            ('financial', Pipeline([
                ('selector', get_fin),
                ('scaler', StandardScaler()),
            ])),

        ],

    )),

    ('classifier', XGBClassifier(seed=7)),
])


pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print(classification_report(y_test, y_pred))