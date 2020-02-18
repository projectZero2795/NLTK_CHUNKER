import nltk
from nltk.chunk.util import *
from nltk.chunk.regexp import *
from nltk import Tree
from nltk import UnigramTagger, BigramTagger, TrigramTagger
from nltk.tag.hmm import HiddenMarkovModelTagger

class Chunker(nltk.ChunkParserI):

    def __init__(self, train_sents):
        train_data = [[(t,c) for w,t,c in sent] for sent in train_sents]
        self.tagger = UnigramTagger(train_data)
        self.tagger = BigramTagger(train_data, backoff= self.tagger)
        self.tagger = TrigramTagger(train_data, backoff=self.tagger)
    
    def parse(self, sentence):
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos), chunktag) in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)