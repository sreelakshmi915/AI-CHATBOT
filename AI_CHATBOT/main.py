import nltk
#nltk.download("punkt")
import numpy as np


from nltk.stem.porter import PorterStemmer
stem_words =PorterStemmer()
def tokenization(sentence):
    return nltk.word_tokenize(sentence)

def stem(words):
    return stem_words.stem(words.lower())



def bow(tokenized_words,all_words):
    tokenized_words = [stem(w)for w in tokenized_words]
    bag = np.zeros(len(all_words),dtype= float32)
    for idx,w in enumerate(all_words):
        if w in tokenized_words:
            bag[idx]=1.0
    return bag




