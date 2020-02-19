import sys
from practica_nltk.src import Tagger, RegexChunker, TrigramChunker, ConsecutiveNPChunker
from practica_nltk.utils import utils
from nltk.corpus import cess_esp

# Constantes

# taggers y chunkers
TAGGER = 'hdm'
REGEX = 'regex'
TRIGRAM = 'trigram'
BAYES = 'bayes'
MODOS = [REGEX, TRIGRAM, BAYES]

# iob tags
CORPUS_PATH = 'practica_nltk/resource/pedidos.txt'
IOB_INPUT_FILE = 'practica_nltk/resource/pedidos_iob_tagged.txt'

# regex config
COMIDA = 'Comida'
CANTIDAD = 'Cantidad'
TO_DETECT_LIST = [COMIDA, CANTIDAD]
CANTIDAD_GRAMMAR = r"""<Z|dn0.*>"""
COMIDA_GRAMMAR = r"""<ncfs.*|aq0.*|sn.*|da0f.*|np.*|sn.*> <sps00>? <di0.*>? <nc.*|aq0.*|sn.*|da0f.*|np.*>?"""

def init_tagger_chunkers():
	# Leer corpus
	sents = cess_esp.tagged_sents()
	corpus = utils.read_corpus(CORPUS_PATH)
	train, test = utils.split_to_train_test_set(corpus)

	# Entrenar tagger
	tagger = Tagger.Tagger(TAGGER, sents)

	# Definir gramática de cantidad y comida
	grammar = {}
	grammar[COMIDA] = COMIDA_GRAMMAR
	grammar[CANTIDAD] = CANTIDAD_GRAMMAR
	regex_chunker = RegexChunker.RegexChunker(grammar)

	# Guardar iob tags
	utils.write_iob_tags(IOB_INPUT_FILE, corpus, tagger, regex_chunker)

	# Leer iob tags y entrenar chunkers
	train_data, test_data = utils.read_iob_tag(IOB_INPUT_FILE)
	trigram_chunker = TrigramChunker.Chunker(train_data, TO_DETECT_LIST)
	bayes_chunker = ConsecutiveNPChunker.ConsecutiveNPChunker(train_data, TO_DETECT_LIST) 

	return tagger, regex_chunker, trigram_chunker, bayes_chunker

def tag(sentence, tagger):
	return tagger.tag(sentence)


# Definir  regex, trigram, y naiveBayes
def chunk(mode, tagged_sentence, regex_chunker, trigram_chunker, bayes_chunker):

	if mode == REGEX:
		return regex_chunker.predict(tagged_sentence)
	elif mode == TRIGRAM:
		return trigram_chunker.predict(tagged_sentence) 
	elif mode == BAYES:
		return bayes_chunker.predict(tagged_sentence)


def instrucciones():
	print("=========== Instrucciones ================")
	print(" (1) Introduce el modo ")
	print(" (2) Introduce la frase a probar")
	print(" (3) Repite (1)")
	print("=========================================\n\n\n")

def pedir_modo():
	print("Introduce el modo ",str(MODOS),":")
	modo = str(input())

	while modo not in MODOS :
		modo = str(input())
	return modo

def pedir_frase():
	print("Introduce la frase a probar:")
	return str(input())


def mostrar_comida_cantidad(dic):
		dif = len(dic[COMIDA]) - len(dic[CANTIDAD])
		if dif > 0: dic[CANTIDAD] += ['1'] * dif
		for key, value in dic.items():
			print(key, ":", str(value))

if __name__ == "__main__":

	# entrenar taggers con el corpus de nltk cess_esp
	# El tagger utilizado es HiddenMarkovModelTagger ya que etiqueta todos los tokens
	tagger, regex_chunker, trigram_chunker, bayes_chunker = init_tagger_chunkers()

	# Mostrar instrucciones
	instrucciones()

	# Programa principal
	while True:

		modo = pedir_modo()
		frase = pedir_frase()

		# Etiqueta cada token 
		tagged_sentence = tag(frase, tagger)
		# Detecta comidas y su cantidad
		dic = chunk(modo, tagged_sentence, regex_chunker, trigram_chunker, bayes_chunker)
		# mostrar resultado
		mostrar_comida_cantidad(dic)