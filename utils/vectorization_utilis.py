#from .preprocessing import spacy_preprocessor
from typing import List,Dict


#def generate_bag_of_words()


def compute_bag_words_sentence(bag_of_words : List[str], sentence : List[str]) -> Dict:
    bag_words=dict.fromkeys(bag_of_words, 0)
    for word in sentence:
        if word in bag_of_words:
            bag_words[word]+=1
    return bag_words


if __name__ == '__main__':
    a=["hi","hello","me"]
    l_doc=["am","hi"]
    print(compute_bag_words_sentence(a,l_doc))

