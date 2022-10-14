from .preprocessing import spacy_preprocessor
from typing import Union, List,Dict,Tuple
import json
import numpy as np
from pathlib import Path

dict_labels={'greeting': 0,
             'goodbye': 1,
             'thanks': 2,
             'hours': 3,
             'mopeds': 4,
             'payments': 5,
             'opentoday': 6,
             'rental': 7,
             'today': 8}
def generate_X_train_Y_train(preprocessor:spacy_preprocessor,json_path:Union[Path, str]) -> Tuple:
    f = open(json_path)
    intents = json.load(f)

    wordlist = []
    X_data = []
    tags=[]

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            pattern = preprocessor.preprocess(pattern)
            if not (pattern == ['']):
                wordlist.extend(pattern)
                X_data.append(pattern)
                tags.append(intent['tag'])

    wordlist = unique(wordlist)
    X_train = np.array([compute_bag_words_sentence(wordlist, pattern) for pattern in X_data])
    Y_train = np.array([dict_labels[tag] for tag in tags])

    return (X_train, Y_train, wordlist)



def unique(list: List) -> List:
    unique_list = []
    for element in list:
        if element not in unique_list:
            unique_list.append(element)
    return unique_list
def compute_bag_words_sentence(bag_of_words : List[str], sentence : List[str]) -> Dict:
    bag_words=dict.fromkeys(bag_of_words, 0)
    for word in sentence:
        if word in bag_of_words:
            bag_words[word]+=1
    return list(bag_words.values())



