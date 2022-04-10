from spacy.lang.en.stop_words import STOP_WORDS as en_stop 
from gensim.models import Phrases
from gensim.corpora.dictionary import Dictionary
import re
import spacy

nlp = spacy.load("en_core_web_md") # think about where best to load this; should not happen at every prediction

def preprocess_text(text, lda_dictionary, phrasecount=1):
  doc=nlp(text.lower()) #get the text
  docs=[]
  for token in doc:
    if not token.like_num:
      if token not in en_stop:
        if len(token) > 1:
          docs.append(token.lemma_)
  docs=[docs] #lemmatize
  bigram = Phrases(docs, min_count=phrasecount) #create bigrams
  for idx in range(len(docs)):#add bigrams
      for token in bigram[docs[idx]]:#token is a bigram
          if '_' in token:#token is a bigram
              # Token is a bigram, add to document.
              docs[idx].append(token)#add bigram

  corpus = [lda_dictionary.doc2bow(doc) for doc in docs] #create corpus
  return corpus

  
def clean(text):
    # standalone sequences of specials, matches &# but not #cool
    text = re.sub(r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)', ' ', text)
    # standalone sequences of hyphens like --- or ==”
    text = re.sub(r'(?:^|\s)[\-=\+]{2,}(?:\s|$)', ' ', text)
    # sequences of white spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
    
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase