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
from sklearn.pipeline import make_pipeline

from mlxtend.classifier import StackingClassifier

from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import TransformerMixin, BaseEstimator


lda_model = pickle.load(open("lda_5.pkl", 'rb'))


def whitespace_tokenizer(s):
   return s.split(' ')


def get_text(X):
    #return [x[0] for x in X[["question"]].values.tolist()]
    return X["question"].tolist()


def get_surf(X):
    return X[["question_sent_count", "question_tok_count", "question_modality_strong", "question_modality_weak", "question_uncertainty", "question_positive", "question_negative"]]


def get_fin(X):
    return X[["VOLA_PRIOR", "VIX", "SIZE", "BTM", "SUE", "INDUSTRY_1", "INDUSTRY_2", "INDUSTRY_3", "INDUSTRY_4",  "INDUSTRY_5", "INDUSTRY_6", "INDUSTRY_7", "INDUSTRY_8", "INDUSTRY_9", "INDUSTRY_10", "INDUSTRY_11", "INDUSTRY_12"]]


class LDATransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return lda_model.transform(data_frame)


df = pd.read_csv("data.csv")
df = pd.get_dummies(df, columns=["INDUSTRY"])

y = df["label"]
X = df.drop(["label"], axis=1)


custom_stop_words = text.ENGLISH_STOP_WORDS.union(["'d","'m","'re","'ve","'s","n't"])
vectorizer = TfidfVectorizer(tokenizer=whitespace_tokenizer, stop_words=custom_stop_words, ngram_range=(1,1))
vectorizer_count = CountVectorizer(tokenizer=whitespace_tokenizer, stop_words=custom_stop_words)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)


## financial features
pipe_fin = make_pipeline(FunctionTransformer(get_fin, validate=False),
						StandardScaler(),
                        XGBClassifier(seed=7)
                        )


## surface features
pipe_surf = make_pipeline(FunctionTransformer(get_surf, validate=False),
						StandardScaler(),
						XGBClassifier(seed=7)
                        )


## lexical features
pipe_text = make_pipeline(FunctionTransformer(get_text, validate=False),
						vectorizer,
						TruncatedSVD(n_components=300, random_state=7),
						StandardScaler(),
						XGBClassifier(seed=7)
                        )


## semantic features
pipe_topic = make_pipeline(FunctionTransformer(get_text, validate=False),
                        vectorizer_count,
                        LDATransformer(),
                        StandardScaler(),
                        XGBClassifier(seed=7)
                        )


sclf = StackingClassifier(classifiers=[pipe_fin, pipe_text, pipe_surf, pipe_topic], meta_classifier=GaussianNB())


sclf.fit(X_train, y_train)

y_pred = sclf.predict(X_test)

print(classification_report(y_test, y_pred))