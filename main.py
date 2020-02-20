import sys
from practica_nltk.src import Tagger, RegexChunker, NGramChunker, ConsecutiveNPChunker
from practica_nltk.utils import utils
from nltk.corpus import cess_esp

# Constantes

# taggers y chunkers
TAGGER = 'hdm'
REGEX = 'regex'
UNIGRAM = 'unigram'
BIGRAM = 'bigram'
TRIGRAM = 'trigram'
BAYES = 'bayes'
MODOS = [REGEX, UNIGRAM, BIGRAM, TRIGRAM, BAYES]

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
INGREDIENTE_GRAMMAR = r"""<sps00> <np.*|nc.*| da0f>"""
COMIDA_GRAMMAR = r"""<aq0.*|sn.*|da0f.*|sn.*|nc.*|Fpt> <Ingrediente>*"""
PEDIDO_GRAMMAR = r"""<Cantidad>? <Comida>"""

def init_tagger_chunkers():
	# Leer corpus
	print("Leyendo cess_esp ...")
	sents = cess_esp.tagged_sents()
	corpus = utils.read_corpus(CORPUS_PATH)
	train, test = utils.split_to_train_test_set(corpus)

	# Entrenar tagger
	print("Entrenando tagger ...")
	tagger = Tagger.Tagger(TAGGER, sents)

	# Definir gramática de cantidad y comida
	print("Inicializando regex chunker ...")
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

	print("Inicializando unigram chunker ...")
	unigram_chunker = NGramChunker.Chunker(train_data, TO_DETECT_LIST)
	print("Inicializando bigram chunker ...")
	bigram_chunker = NGramChunker.Chunker(train_data, TO_DETECT_LIST, 2)
	print("Inicializando trigram chunker ...")
	trigram_chunker = NGramChunker.Chunker(train_data, TO_DETECT_LIST, 3)
	print("Inicializando bayes chunker ...")
	bayes_chunker = ConsecutiveNPChunker.ConsecutiveNPChunker(train_data, TO_DETECT_LIST) 

	chunkers = {REGEX: regex_chunker, UNIGRAM: unigram_chunker, BIGRAM: bigram_chunker, TRIGRAM: trigram_chunker, BAYES: bayes_chunker}
	return tagger, chunkers

def tag(sentence, tagger):
	return tagger.tag(sentence)

def parse(chunked):
	root = chunked[0]['S']
	pedido = {}
	pedidos = []
	pedido_completo = False
	for element in root:
		# si es un pedido
		if isinstance(element, dict):
			if pedido_completo:
				
				# Puede pasar que la comida ya se ha detectado pero se desconoce la cantidad
				# Por defecto, tendrá un valor uno
				if CANTIDAD not in element:
					pedido[CANTIDAD] = '1'

				# En todos los pedidos el ingrediente es el último en mencionar y es opcional
				if INGREDIENTE in element:
					pedido[INGREDIENTE] = ' '.join(element[INGREDIENTE])

				pedidos.append(pedido)
				pedido = {}

			if PEDIDO in element:
				for x in element[PEDIDO]:
					# puede ser cantidad o comida
					if isinstance(x, dict):
						# Si es un nodo cantidad
						if CANTIDAD in x:
							pedido[CANTIDAD] = x[CANTIDAD]
						#Si es un nodo comida
						elif COMIDA in x:
							# recorremos el arbol
							for y in x[COMIDA]:
								# si es un nodo hoja es comida
								if isinstance(y, str):
									pedido[COMIDA] = y
								# si no es un nodo hoja es ingrediente
								elif isinstance(y, dict):
									pedido[INGREDIENTE] = ' '.join(y[INGREDIENTE])

			#### NOTA: Puede pasar que el chunker no pueda detectar un pedido
			####	   pero si que deteca comida, cantidad, y ingredientes

			# Si es un nodo cantidad
			if CANTIDAD in element:
				pedido[CANTIDAD] = element[CANTIDAD]
			#Si es un nodo comida
			elif COMIDA in element:
				pedido[COMIDA] = element[COMIDA]
			# Si es un nodo element
			elif INGREDIENTE in element:
				pedido[INGREDIENTE] = element[INGREDIENTE]

			# Decimos que un pedido es completo cuando ya hemos detectado la comida
			# En este caso la cantidad y sus ingredientes son opcionales
			# Podemos cambiar las reglas añadiendo más condiciones
			pedido_completo = COMIDA in pedido # and CANTIDAD in pedido

	if pedido_completo:
		# Puede pasar que la comida ya se ha detectado pero se desconoce la cantidad
		# Por defecto, tendrá un valor uno
		if CANTIDAD not in element:
			pedido[CANTIDAD] = '1'
		# En todos los pedidos el ingrediente es el último en mencionar y es opcional
		if INGREDIENTE in element:
			pedido[INGREDIENTE] = ' '.join(element[INGREDIENTE])
		pedidos.append(pedido)

	return pedidos

def nueva_linea():
	print("\n")

def mostrar_pedidos(pedidos):

	for pedido in pedidos:
		print(str(pedido))

def mostrar_tagged_sents(tagged_sents):
	print(str(tagged_sents))

def chunk(mode, tagged_sentence, chunkers):

	if mode in chunkers:
		prediction = chunkers[mode].predict(tagged_sentence)
		return parse(prediction)


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


if __name__ == "__main__":

	# entrenar taggers con el corpus de nltk cess_esp
	# El tagger utilizado es HiddenMarkovModelTagger ya que etiqueta todos los tokens
	tagger, chunkers = init_tagger_chunkers()

	# Mostrar instrucciones
	instrucciones()

	# Programa principal
	while True:

		modo = pedir_modo()
		frase = pedir_frase()

		nueva_linea()
		# Etiqueta cada token 
		tagged_sentence = tag(frase, tagger)
		print("tags: ")
		mostrar_tagged_sents(tagged_sentence)
		nueva_linea()

		# Detecta comidas y su cantidad
		pedidos = chunk(modo, tagged_sentence, chunkers)
		print("pedidos: ")
		# mostrar resultado
		mostrar_pedidos(pedidos)
		nueva_linea()