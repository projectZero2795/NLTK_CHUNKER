import sys
import nltk
from practica_nltk.src import Tagger, RegexChunker, TrigramChunker, ConsecutiveNPChunker
from practica_nltk.utils import utils
from nltk.corpus import cess_esp



# Constantes
# Modos
TAGGER = 'hdm'
REGEX = 'regex'
TRIGRAM = 'trigram'
BAYES = 'bayes'


# iob tags
CORPUS_PATH = 'practica_nltk/resource/pedidos.txt'
IOB_INPUT_FILE = 'practica_nltk/resource/pedidos_iob_tagged.txt'

# regex config
COMIDA = 'Comida'
INGREDIENTE = 'Ingrediente'
CANTIDAD = 'Cantidad'
PEDIDO = 'Pedido'
TO_DETECT_LIST = [PEDIDO, CANTIDAD, COMIDA, INGREDIENTE]
CANTIDAD_GRAMMAR = r"""<Z|dn0.*>"""
INGREDIENTE_GRAMMAR = r"""<sps00> <np.*|nc.*>"""
COMIDA_GRAMMAR = r"""<aq0.*|sn.*|da0f.*|sn.*|nc.*|Fpt> <Ingrediente>*"""
PEDIDO_GRAMMAR = r"""<Cantidad>? <Comida>"""

def init_tagger_chunkers():
	# Leer corpus
	sents = cess_esp.tagged_sents()
	corpus = utils.read_corpus(CORPUS_PATH)
	train, test = utils.split_to_train_test_set(corpus)

	# Entrenar tagger
	tagger = Tagger.Tagger(TAGGER, sents)

	# Definir gramática de cantidad y comida
	grammar = {}
	grammar[INGREDIENTE] = INGREDIENTE_GRAMMAR
	grammar[COMIDA] = COMIDA_GRAMMAR
	grammar[CANTIDAD] = CANTIDAD_GRAMMAR
	grammar[PEDIDO] = PEDIDO_GRAMMAR
	regex_chunker = RegexChunker.RegexChunker(grammar)

	# Guardar iob tags
	utils.write_iob_tags(IOB_INPUT_FILE, corpus, tagger, regex_chunker)

	# Leer iob tags
	train_data, test_data = utils.read_iob_tag(IOB_INPUT_FILE)

	trigram_chunker = TrigramChunker.Chunker(train_data, TO_DETECT_LIST)
	bayes_chunker = ConsecutiveNPChunker.ConsecutiveNPChunker(train_data, TO_DETECT_LIST) 
	return regex_chunker, tagger, trigram_chunker, bayes_chunker, test_data

def tag(sentence, tagger):
	return tagger.tag(sentence)

def mostrar_comida_ingrediente_cantidad(dic):
		root = dic[0]['S']
		for element in root:
			# si es un pedido
			if isinstance(element, dict):
				for x in element[PEDIDO]:
					# puede ser cantidad o comida
					if isinstance(x, dict)
		#dif = len(dic[COMIDA]) - len(dic[CANTIDAD])
		#if dif > 0: dic[CANTIDAD] += ['1'] * dif
		#for key, value in dic.items():
		#	print(key, ":", str(value))

def chunk(tagged_sentence, regex_chunker, trigram_chunker, bayes_chunker, test_data):
	print("Testing regex chunker ...")
	mostrar_comida_cantidad(regex_chunker.predict(tagged_sentence))
	#print(regex_chunker.evaluate(test_data))
	#print("Testing trigram chunker ...")
	#mostrar_comida_cantidad(trigram_chunker.predict(tagged_sentence))
	#print(trigram_chunker.evaluate(test_data))
	#print("Testing bayes chunker ...")
	#mostrar_comida_cantidad(bayes_chunker.predict(tagged_sentence)) 
	#print(bayes_chunker.evaluate(test_data))

if __name__ == "__main__":
	test_sentence = "quiero 3 bocadillos de anchoas, 2 pizzas, 2 tortilla de patatas"
	if len(sys.argv) > 1:
		test_sentence = sys.argv[1]

	print("Frase a probar: ",test_sentence)

	regex_chunker, tagger, trigram_chunker, bayes_chunker, test_data = init_tagger_chunkers()
	tagged_sentence = tag(test_sentence, tagger)	
	print("Tagged : ",str(tagged_sentence))
	chunk(tagged_sentence, regex_chunker, trigram_chunker, bayes_chunker, test_data)