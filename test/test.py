import sys
from practica_nltk.src import Tagger, RegexChunker, TrigramChunker, ConsecutiveNPChunker
from practica_nltk.utils import utils
from nltk.corpus import cess_esp

# Constantes
TAGGER = 'hdm'
REGEX = 'regex'
TRIGRAM = 'trigram'
CONSECUTIVE_NP_CHUNKER = 'consec'
TEST_SENTENCE = "Quiero 1 pizza de pepperoni con extra de mozarella, y de bebida, una coca-cola."
IOB_INPUT_FILE = 'pedidos_iob_tagged.txt'

# Leer corpus
sents = cess_esp.tagged_sents()
corpus = utils.read_corpus('practica_nltk/resource/pedidos.txt')
train, test = utils.split_to_train_test_set(corpus)

# Entrenar tagger
tagger = Tagger.Tagger(TAGGER, sents)
tagged_sentence = tagger.tag(TEST_SENTENCE)

# Definir gramática de cantidad y comida
grammar = {}
grammar['Cantidad'] = r"""<Z|dn0.*>"""
grammar['Comida'] = r"""<ncfs.*|aq0.*|sn.*|da0f.*|np.*|sn.*> <sps00>? <di0.*>? <nc.*|aq0.*|sn.*|da0f.*|np.*>?"""
regex_chunker = RegexChunker.RegexChunker(grammar)

# Guardar iob tags
utils.write_iob_tags(IOB_INPUT_FILE, corpus, tagger, regex_chunker)

# Leer iob tags
train_data, test_data = utils.read_iob_tag(IOB_INPUT_FILE)

# Definir tests regex, trigram, y naiveBayes
def test_regex_chunker():
	print("Testing regex chunker ...")
	regex_chunker.predict_and_print(TEST_SENTENCE)

def test_trigram_chunker(train, tagged_sentence):
	print("Testing trigram chunker ...")
	chunker = TrigramChunker.Chunker(train)
	chunker.parse(tagged_sentence) 

def test_consecutive_np_chunker(train, tagged_sentence):
	print("Testing test_consecutive_np_chunker ...")
	chunker = ConsecutiveNPChunker.ConsecutiveNPChunker(tagged_sentence) 
	chunker.parse(tagged_sentence)


if __name__ == "__main__":
	if not len(sys.argv):
		test_regex_chunker()
		test_trigram_chunker(train, tagged_sentence)
		test_consecutive_np_chunker(train, tagged_sentence)
	else:
		for mode in sys.argv:
			if mode == REGEX:
				test_regex_chunker()
			elif mode == TRIGRAM:
				test_trigram_chunker(train, tagged_sentence)
			elif mode == CONSECUTIVE_NP_CHUNKER:
				test_consecutive_np_chunker(train, tagged_sentence)
			else:
				print("Error (",mode,"):- Posibles argumentos:",REGEX, " , ",TRIGRAM, " , ",CONSECUTIVE_NP_CHUNKER)


