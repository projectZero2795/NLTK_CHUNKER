import nltk


def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    if i == 0:
       prevword, prevpos = "<START>", "<START>"
    else:
       prevword, prevpos = sentence[i-1]
    return {"pos": pos, "prevpos": prevpos}

class ConsecutiveNPChunkTagger(nltk.TaggerI):
    
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i ,history)
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)
    
    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI):
    
    def __init__(self, train_sents, to_detect_list):
        tagged_sents = [[((w,t), c) for (w,t,c) in sent]
                           for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)
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


    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)
            
    def predict(self, sentence):
        chunked_sentence = self.parse(sentence)
        dic = []
        self.traverse_to_dic(chunked_sentence, dic)
        return dic