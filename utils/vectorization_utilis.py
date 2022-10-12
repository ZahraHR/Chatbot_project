from .preprocessing import spacy_preprocessor
from typing import Union, List,Dict,Tuple
import json
import numpy as np

#preprocessor = spacy_preprocessor()
def generate_X_train_Y_train(preprocessor:spacy_preprocessor,json_path:Union[int, str]) -> Tuple:
    f = open(json_path)
    intents = json.load(f)

    wordlist = []
    all_data = []

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            pattern = preprocessor.preprocess(pattern)
            if not (pattern == ['']):
                wordlist.extend(pattern)
                all_data.append((pattern, intent['tag']))
    wordlist = unique(wordlist)

    X_train = np.array([compute_bag_words_sentence(wordlist, pattern) for (pattern, tag) in all_data])
    Y_train = np.array([tag for (pattern, tag) in all_data])

    return(X_train,Y_train)
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


if __name__ == '__main__':
    a=["hi","hello","me"]
    l_doc=["am","hi"]
    print(compute_bag_words_sentence(a,l_doc))

