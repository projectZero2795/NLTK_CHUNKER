from nltk import UnigramTagger, BigramTagger, TrigramTagger
from nltk.tag.hmm import HiddenMarkovModelTagger


TRIGRAM = 'trigram'
HDM = 'hdm'
class Tagger(object):
	def __init__(self, mode, train_sents):
		if mode == TRIGRAM:
			self.tagger = UnigramTagger(train_sents)
			self.tagger = BigramTagger(train_sents, backoff= self.tagger)
			self.tagger = TrigramTagger(train_sents, backoff=self.tagger)
		else:
			self.tagger = HiddenMarkovModelTagger.train(train_sents)
			
	def tag(self, sentence):
		sentence_tokens = nltk.word_tokenize(sentence)
		return self.tagger.tag(sentence_tokens)