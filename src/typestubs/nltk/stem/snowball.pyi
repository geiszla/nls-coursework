from nltk.stem.api import StemmerI

class SnowballStemmer(StemmerI):
    def __init__(self, language: str, ignore_stopwords: bool = False):
        ...

    def stem(self, token: str) -> str:
        ...
