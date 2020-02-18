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
            feautureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentences, history)

class ConsecutiveNPChunker(nltk.ChunkParserI):
    
    def __init__(self, train_sents):
        tagged_sents = [[((w,t), c) for (w,t,c) in nltk.chunk.tree2conlltags(sent)]
                           for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)
            
    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)