from typing import List
import nltk
import string
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

import spacy

class nltk_prepocessor:

    def __init__(self):
        self.stopwords = stopwords.words('english')
        self.porter_stemmer = PorterStemmer()

    def remove_punctuation(self,sentence:str) -> str:
        self.sentence = sentence
        sentence = "".join([i for i in self.sentence if i not in string.punctuation])
        return str(sentence)

    def lower_text(self,sentence:str) -> str:
        self.sentence = sentence
        return str(self.sentence.lower())

    def remove_stopwords_from_tokens(self,sentence:str) -> List[str]:
        self.sentence = sentence
        return [str(token) for token in self.sentence.split(" ") if token not in self.stopwords]

    def stemming(self, tokens: List[str]) -> List[str]:
        self.tokens=tokens
        return [self.porter_stemmer.stem(token) for token in tokens]

    def preprocess(self,sentence: str) -> List[str]:
        #self.sentence = sentence
        tokens = self.remove_punctuation(sentence)
        tokens = self.lower_text(tokens)
        tokens = self.remove_stopwords_from_tokens(tokens)
        return self.stemming(tokens)


class spacy_preprocessor:
    def __init__(
            self,
            spacy_model=None,
            remove_numbers=True,
            remove_special=True,
            remove_unwanted_tokens =True,
            remove_stopwords=True,
            lemmatization=True,
            stemming=False,

    ):
        self._remove_numbers = remove_numbers
        self._remove_stopwords = remove_stopwords
        self._remove_special = remove_special
        self.remove_unwanted_tokens = remove_unwanted_tokens
        self._lemmatization = lemmatization
        self._stemming = stemming

        self.porter_stemmer = PorterStemmer()

        if not spacy_model:
            try:
                self.model = spacy.load("en_core_web_sm")
            except:
                self.model=self._download_model("en_core_web_sm")
        else:
            self.model = self._download_model(spacy_model)

    def _download_model(self,model_str:str):
        spacy.cli.download(model_str)
        return spacy.load(model_str)

    def preprocess(self,sentence:str) -> List[str]:

        tokens = self._clean(sentence)

        if self._lemmatization:
            tokens = [token.lemma_ for token in tokens]
        else:
            tokens = [token.text for token in tokens]

        if self._stemming:
            [self.porter_stemmer.stem(token) for token in tokens]

        text = " ".join([token for token in tokens])

        text = text.lower()

        return text.split(" ")

    def _clean(self, sentence:str) -> List:

        sentence = re.sub(r"[^a-zA-Z\']", " ", sentence)
        tokens=[token for token in self.model(sentence)]

        if self._remove_numbers:
            tokens = [token for token in tokens if not token.like_num]
        if self._remove_stopwords:
            tokens = [token for token in tokens if not token.is_stop]
        if self.remove_unwanted_tokens:
            tokens = [token for token in tokens if not (token.is_punct or token.is_space or token.is_quote or token.is_bracket or token.is_currency)]
        return tokens

