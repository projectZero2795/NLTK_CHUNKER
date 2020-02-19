import nltk
from nltk import RegexpParser

class RegexChunker(nltk.ChunkParserI):

	def __init__(self, grammar):

		self.grammar = grammar
		self.init_parser()

	def set_grammar(self, grammar):
		self.grammar = grammar

	def init_parser(self):
		grammar = ""
		for key, value in self.grammar.items():
			grammar = grammar  + key + ": {" + value + "} \n"
		self.parser = RegexpParser(grammar)

	def parse(self, sentence):
		return  self.parser.parse(sentence)


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

	def predict(self, tagged_sentence):
		chunked_sentence = self.parser.parse(tagged_sentence)
		dic = []
		self.traverse_to_dic(chunked_sentence, dic)
		return dic

	


		