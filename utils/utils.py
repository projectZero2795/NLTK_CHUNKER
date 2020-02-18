from practica_nltk.resource import *

def read_corpus(input_file):
	f = open(input_file, 'r')
	corpus = f.readlines()
	f.close()
	return corpus

def split_to_train_test_set(tagged_sents):
	train = []
	test = []
	for i in range(len(tagged_sents)):
		if i % 10:
			train.append(tagged_sents[i])
		else:
			test.append(tagged_sents[i])
	return train, test


def read_iob_tag(input_file):
    f = open(input_file, 'r')
    corpus_iob = f.readlines()
    f.close()

    train = []
    test = []
    sent = []
    for line in corpus_iob:
        data = line.split()
        if len(data):
            word = data[0]
            tag = data[1]
            chunk_tag = data[2]
            word_tagged = (word, tag, chunk_tag)
            sent.append(word_tagged)
        else:
            if i % 10:
                train.append(sent)
            else:
                test.append(sent)
            sent = []
    return train, test

def traverse(t, iob, label, f):
    try:
        t.label()
    except AttributeError:
        if label != 'S':
            if iob == 1:
                f.write(' '.join(list(t)) + " B-"+label)
                iob = 2
            elif iob == 2:
                f.write(' '.join(list(t)) +  " I-"+label)
            else:
                f.write(' '.join(list(t))+" O ")
        else:
            f.write(' '.join(list(t))+" O ")
        f.write("\n")
    else:
        iob = 1
        for child in t:
            iob = traverse(child, iob, t.label(), f)
        iob = 0
    return iob


def write_iob_tags(output_file, corpus, regex_chunker):
    f = open(output_file, 'w')
    for sentence in corpus:
        sentence_tokens = nltk.word_tokenize(sentence)
        tagged_sentence = regex_chunker.tagger.tag(sentence_tokens)
        chunked_sentence = regex_chunker.parser.parse(tagged_sentence)
        traverse(chunked_sentence, 0, "", f)
        f.write("\n")
    f.close()

