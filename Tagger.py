import sys

import numpy as np

from scipy.sparse import csr_matrix,lil_matrix
from scipy.sparse import coo_matrix
from Data import LinearChainData

class Tagger(object):
    def __init__(self, average=True):
        self.useAveraging = average

    def ComputeThetaAverage(self):
        self.thetaAverage = self.theta - (self.thetaSum/float(self.nUpdates))

    def PrintableSequence(self, sequence):
        return [self.train.tagVocab.GetWord(x) for x in sequence]

    def DumpParameters(self, outFile):
        fOut = open(outFile, 'w')
        sortedParams = (np.argsort(self.thetaAverage, axis=None)[::-1])[0:500]
        for i in sortedParams:
            (tag1ID, tag2ID, featureID) = np.unravel_index(i, self.theta.shape)
            fOut.write("%s %s %s %s\n" % (self.train.tagVocab.GetWord(tag1ID), self.train.tagVocab.GetWord(tag2ID), self.train.vocab.GetWord(featureID), self.thetaAverage[tag1ID,tag2ID,featureID]))
        fOut.close()

    def Train(self, nIter):
        nSent = 0
        self.u = self.theta
        for i in range(nIter):
            for (s,g) in self.train.featurizedSentences:
                if len(g) <= 1:         #Skip any length 1 sentences - some numerical issues...
                    continue
                z = self.Viterbi(s, self.theta, len(g))

                sys.stderr.write("Iteration %s, sentence %s\n" % (i, nSent))
                sys.stderr.write("predicted:\t%s\ngold:\t\t%s\n" % (self.PrintableSequence(z), self.PrintableSequence(g)))
                nSent += 1                
                self.UpdateTheta(s,g,z, self.theta, len(g))
        if self.useAveraging:
            self.ComputeThetaAverage()

class ViterbiTagger(Tagger):
    def __init__(self, inFile, average=True):
        self.train = LinearChainData(inFile)
        self.useAveraging = average

        self.ntags    = self.train.tagVocab.GetVocabSize()
        self.theta    = np.zeros((self.ntags, self.ntags, self.train.vocab.GetVocabSize()))   #T^2 parameter vectors (arc-emission CRF)
        self.thetaSum = np.zeros((self.ntags, self.ntags, self.train.vocab.GetVocabSize()))   #T^2 parameter vectors (arc-emission CRF)
        self.nUpdates = 0

    def TagFile(self, testFile):
        self.test = LinearChainData(testFile, vocab=self.train.vocab)
        for i in range(len(self.test.sentences)):
            featurizedSentence = self.test.featurizedSentences[i][0]
            sentence = self.test.sentences[i]
            if self.useAveraging:
                v = self.Viterbi(featurizedSentence, self.thetaAverage, len(sentence))
            else:
                v = self.Viterbi(featurizedSentence, self.theta, len(sentence))
            words = [x[0] for x in sentence]
            tags  = self.PrintableSequence(v)
            for i in range(len(words)):
                print "%s\t%s" % (words[i], tags[i])
            print ""

    def Viterbi(self, featurizedSentence, theta, slen):
        """Viterbi"""
        #TODO: Implement the viterbi algorithm (with backpointers)
        slen += 1
        viterbiSeq = [0 for i in range(slen)]
        alpha = [[-1 for j in range(self.ntags)] for i in range(slen)]
        tau = [[-1 for j in range(self.ntags)] for i in range(slen)]
        alpha[0][0] = 1

        for i in range(slen-1):
            for j in range(self.ntags):
                h = featurizedSentence.getrow(i).dot(theta[:,j].T) + alpha[i]
                alpha[i+1][j] = np.max(h)
                tau[i+1][j] = np.argmax(h)

        viterbiSeq[slen-1] = np.argmax(alpha[slen-1])
        for i in range(slen-2,-1,-1):
            viterbiSeq[i] = tau[i+1][viterbiSeq[i+1]]
            #print tau[viterbiSeq[i+1]][i+1]


        #viterbiSeq = [self.train.tagVocab.GetID('NOUN') for x in range(featurizedSentence.shape[0])]
        return viterbiSeq[1:]


    #Structured Perceptron update
    def UpdateTheta(self, sentenceFeatures, 
                          goldSequence, 
                          viterbiSequence,
                          theta,
                          slen):
        
        ntags = self.ntags
        START_TAG = self.train.tagVocab.GetID('START')
        nFeatures = self.train.vocab.GetVocabSize()
        gseq = [START_TAG] + list(goldSequence)
        vseq = [START_TAG] + viterbiSequence

        for i in range(0,slen-1):
            if not (gseq[i+1] == vseq[i+1] and gseq[i] == vseq[i]):
                self.theta[gseq[i]][gseq[i+1]] += sentenceFeatures.getrow(i)
                self.theta[vseq[i]][vseq[i+1]] -= sentenceFeatures.getrow(i)
                self.nUpdates += 1
                self.thetaSum[gseq[i]][gseq[i+1]] += sentenceFeatures.getrow(i)*self.nUpdates
                self.thetaSum[vseq[i]][vseq[i+1]] -= sentenceFeatures.getrow(i)*self.nUpdates

