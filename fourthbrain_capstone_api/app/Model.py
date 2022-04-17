from gensim.test.utils import common_texts, datapath
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from pydantic import BaseModel

from app.helperfunctions import clean, decontracted, preprocess_text

from spacy.lang.en.stop_words import STOP_WORDS as en_stop 
en_stop.add("said") # add 'said to stop words because it appears a lot in the data


class InputDoc(BaseModel):
    input_doc: str


class MyLdaModel(object):

    def __init__(self):
        self.model_fname_ = 'app/api_model/LDAModel'
        self.corpus_dict_fname_ = 'app/api_model/dict'
        # load spacy module here..
        try:
            self.model = LdaModel.load(self.model_fname_)
            self.common_dict = Dictionary.load(self.corpus_dict_fname_)
        except Exception as _:
            pass # bad practice

    def predict_topics(self, input_doc):
        # Preprocess text
        data_in = input_doc

        data_in=clean(data_in) #clean text
        data_in=decontracted(data_in)#decontract text
        corpus_gensim = preprocess_text(data_in, self.common_dict, phrasecount=1)

        cluster_names = {#name the clusters as seems reasonable
            0: "Technology/Service/Industry/Business",
            1: "Law/Policing/State",
            2: "Politics",
            3: "Art/Tech/Lifestyle",
            4: 'World News/Trade/Politics',
            5: 'Health/Misc.'
        }
        
        unseen_doc = corpus_gensim[0] #get unseen document
        vector = self.model[unseen_doc]  # get topic probability distribution for a document
        topic_number, proba = sorted(vector, key=lambda item: item[1])[-1] #get the most probable topic

        if proba < 0.2: # not a relevant topic
            return "no matching topic", 0.00
        else:
            return cluster_names.get(topic_number), proba


if __name__ == '__main__':
    my = MyLdaModel()
    res = my.predict_topics('computer graph time horizon')
    print(res)
