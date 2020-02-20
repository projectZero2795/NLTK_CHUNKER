import nltk
from nltk.chunk.util import *
from nltk.chunk.regexp import *
from nltk import Tree
from nltk import UnigramTagger, BigramTagger, TrigramTagger
from nltk.tag.hmm import HiddenMarkovModelTagger

class Chunker(nltk.ChunkParserI):

    def __init__(self, train_sents, to_detect_list, n_gram = 1):
        train_data = [[(t,c) for w,t,c in sent] for sent in train_sents]

        self.tagger = UnigramTagger(train_data)
        if n_gram > 1:
            self.tagger = BigramTagger(train_data, backoff= self.tagger)
        if n_gram > 2:
            self.tagger = TrigramTagger(train_data, backoff=self.tagger)
        self.to_detect_list = to_detect_list

    def traverse_to_dic(self, t, dicc):
        try:
            t.label()
        except AttributeError:
            dicc.append(list(t)[0])
        else:
            new_list = []
            new_dicc = {t.label():new_list}
            dicc.append(new_dicc)
            for child in t:
                self.traverse_to_dic(child, new_list)

        return None

    def parse(self, sentence):
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos), chunktag) in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)
    
    def predict(self, sentence):
        chunked_sentence = self.parse(sentence)
        dic = []
        self.traverse_to_dic(chunked_sentence, dic)
        return dic