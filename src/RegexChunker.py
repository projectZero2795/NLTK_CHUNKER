from nltk import RegexpParser

class RegexChunker(object):

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



	def traverse_to_dic(t, dicc, label = ""):
	    try:
	        t.label()
	    except AttributeError:
	        if label in dicc.keys():
	        	dicc[label].append(list(t)[0])
	    else:
	        for child in t:
	            traverse_to_dic(child, dicc, t.label())

	    return None

	def predict(self, tagged_sentence):
		chunked_sentence = self.parser.parse(tagged_sentence)
		dic = {}
		for key, _ in self.grammar:
			dic[key] = []

		traverse_to_dic(chunked_sentence, dic)
		return dic

	def predict_and_print(self, tagged_sentence):
		chunked_sentence = self.parser.parse(tagged_sentence)
		dic = {}
		for key, _ in self.grammar:
			dic[key] = []

		traverse_to_dic(chunked_sentence, dic)
		
		for key, value in dic:
			print(key, ":", str(value))


		