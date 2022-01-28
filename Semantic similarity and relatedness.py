# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 12:19:56 2020

@author: 49628
"""
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import spacy
import numpy as np
# Get the interactive Tools for Matplotlib
#%matplotlib notebook
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.decomposition import PCA

import codecs
import json
import os
import sys
import argparse
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Gensim imports
from gensim.test.utils import datapath
from gensim import utils
import gensim.models
import gensim.downloader as api


###############################################################################
# Define a Corpus object that has an iterator
# that will be used by gensim to access the data
# in the corpus line by line
class Corpus(object):
    """ Corpus object Initalizer """
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path

    """An iterator that yields sentences (lists of str)."""
    def __iter__(self):
        # corpus_path = "./corpora/MTSamplesPlainText.txt"
        for line in codecs.open(self.corpus_path, "r", encoding="utf-8", errors='ignore'):
            # assume there's one document per line, tokens separated by whitespace
            yield preprocess_text(line)

###############################################################################
# Function needed to recursively access values inside the CORD-19 corpus JSON
# files to pull out the text fields
def extract_values(obj, key):
    """Pull all values of specified key from nested JSON."""
    arr = []

    def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    results = extract(obj, arr, key)
    return results

###############################################################################
# do some text normalization (tokenization, etc.)
def preprocess_text(txt_in):
    # TODO: this is where you want to write your own code
    # to manipulate input text

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    # ---------------------------------------------------------------- #
    # !YOUR CODE for pre-processing HERE
    # a) tokenize the text in "txt_in" variable (√)
    # b) tag each token for part-of-speech (√)
    # c) select a subset of words to be appended to "txt_out"
    # (e.g., the subset could be all nouns, all verbs, etc.)(√)
    # ---------------------------------------------------------------- #
    # Extra processing 
    # d) lemmanization
    # e) lowcase everything
    # f) remove stop words

    # below is a call to a utility function in Gensim that will return a list of tokens
    # using a simple whitespace tokenizer
    # if you wish to use your own tokenization, you can REPLACE this line
    # with your own code - just make sure that txt_out is a list of tokens
    # you can use any packages or utilities (e.g., NLTK, scikit-learn, SpaCY, etc.
   
#    sp = spacy.load('en')
#    def lemma_function(text):
#        dummy = []
#        #this is just a test to see if it works
#        for word in sp(text):
#            dummy.append(word.lemma_)
#            return ' '.join(dummy)
#    lemma = lemma_function(txt_in)
    tokens = utils.simple_preprocess(txt_in)
    txt_out = nltk.pos_tag(tokens)
    #Tag each token, txt_out is a list of tuple (token, tag)
    is_noun = lambda pos: pos[:2] == 'NN'
    is_verb = lambda pos: pos[:2] == 'VB'
    # do the nlp stuff
    txt_out_nv = [word for (word, pos) in txt_out if is_noun(pos) or is_verb(pos)] 
    txt_out = [x for x in txt_out_nv if x not in stopwords.words('english') ] 
    return txt_out


###############################################################################
# function to prepare the CORD-19 corpus

def prepare_cord19(cord19dir, cord19corpus):

    cor = open(cord19corpus,"w", encoding = 'utf-8')
    fcnt = 0
    for subdir, dirs, files in os.walk(cord19dir):
        for file in files:
            #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file

            if filepath.endswith(".json"):
                fcnt = fcnt + 1

                with open(filepath) as json_file:
                    data = json.load(json_file)
                    #text = map(lambda datum: datum['text'], data)

                    for txt in extract_values(data['abstract'],'text') + extract_values(data['body_text'],'text'):
                        # write out pre-processed text to the corpus file
                        # clean citations
                        txt = re.sub("\[\d+\]", ' ', txt)
                        cor.write(txt+"\n")
                        sys.stdout.write("Processed progress: %d files  \r" % (fcnt))
                        sys.stdout.flush()


    cor.close()

###############################################################################
#function to visualize word2vec models 
def display_pca_scatterplot(model, words=None, sample=0):
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            words = [ word for word in model.vocab ]
            
    word_vectors = np.array([model[w] for w in words])
    
    twodim = PCA().fit_transform(word_vectors)[:,:2]
        
    plt.figure(figsize=(6,6))
    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
    for word, (x,y) in zip(words, twodim):
        plt.text(x+0.05, y+0.05, word)


if __name__== "__main__":

    # process arguments to the script
    parser = argparse.ArgumentParser(description="My version of run_word2vec.py script")
    parser.add_argument("--output", default="C:/Users/49628/Desktop/UMN/2020 Spring/BioNLP/HW2/output.log", help="Output file containing results.")
    args = parser.parse_args()
    
    # open the output log file for writing
    outputfile = open(args.output, "w")

    # prepare CORD19 corpus
#    cord19dir = 'C:/Users/49628/Desktop/UMN/2020 Spring/BioNLP/HW2/'
#    cord19corpus = 'C:/Users/49628/Desktop/UMN/2020 Spring/BioNLP/HW2/CORD-19.cor'
#    prepare_cord19(cord19dir, cord19corpus)



    # train CORD19 word2vec model
    sentences = Corpus("C:/Users/49628/Desktop/UMN/2020 Spring/BioNLP/HW2/CORD-19.cor")
    # Use CBOW method
    cord19model_CBOW = gensim.models.Word2Vec(sentences=sentences, size=300, sg=0)
    # Use SG method
    cord19model_SG = gensim.models.Word2Vec(sentences=sentences, size=300, sg=1)


    # run Pearson and spearman rank correlations with human ratings
    outputfile.write("\n========================= CORD19 UMNSRS_similarity =================\n")
    outputfile.write("\n========================= Use CBOW method =================\n")    
    pearson, spearman, oovr = cord19model_CBOW.wv.evaluate_word_pairs('C:/Users/49628/Desktop/UMN/2020 Spring/BioNLP/HW2/UMNSRS_similarity.tsv')
    outputfile.write("Perason R: "+str(pearson[0])+"\n")
    outputfile.write("Spearman Rho: " + str(spearman[0]) +"\n")
    outputfile.write("OOV rate: " + str(oovr))
    outputfile.write("\n====================================================================\n")
    outputfile.write("\n========================= Use SG method =================\n")        
    pearson, spearman, oovr = cord19model_SG.wv.evaluate_word_pairs('C:/Users/49628/Desktop/UMN/2020 Spring/BioNLP/HW2/UMNSRS_similarity.tsv')
    outputfile.write("Perason R: "+str(pearson[0])+"\n")
    outputfile.write("Spearman Rho: " + str(spearman[0]) +"\n")
    outputfile.write("OOV rate: " + str(oovr))
    outputfile.write("\n====================================================================\n")


    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    # ---------------------------------------------------------------- #
    # ADD YOUR CODE for evaluation on UMNSRS_relatedenss HERE ##############
    # ---------------------------------------------------------------- #
    outputfile.write("\n========================= CORD19 UMNSRS_relatedness =================\n")
    outputfile.write("\n========================= Use CBOW method =================\n")    
    pearson, spearman, oovr = cord19model_CBOW.wv.evaluate_word_pairs('C:/Users/49628/Desktop/UMN/2020 Spring/BioNLP/HW2/UMNSRS_relatedness.tsv')
    outputfile.write("Perason R: "+str(pearson[0])+"\n")
    outputfile.write("Spearman Rho: " + str(spearman[0]) +"\n")
    outputfile.write("OOV rate: " + str(oovr))
    outputfile.write("\n====================================================================\n")
    outputfile.write("\n========================= Use SG method =================\n")        
    pearson, spearman, oovr = cord19model_SG.wv.evaluate_word_pairs('C:/Users/49628/Desktop/UMN/2020 Spring/BioNLP/HW2/UMNSRS_relatedness.tsv')
    outputfile.write("Perason R: "+str(pearson[0])+"\n")
    outputfile.write("Spearman Rho: " + str(spearman[0]) +"\n")
    outputfile.write("OOV rate: " + str(oovr))
    outputfile.write("\n====================================================================\n")




    # train MTSamples word2vec model
    sentences = Corpus("C:/Users/49628/Desktop/UMN/2020 Spring/BioNLP/HW2/MTSamplesPlainText.txt")
    # Use CBOW method
    MTSmodel_CBOW = gensim.models.Word2Vec(sentences=sentences, size=300, sg=0)
    # Use SG method
    MTSmodel_SG = gensim.models.Word2Vec(sentences=sentences, size=300, sg=1)
    
    outputfile.write("\n========================= MTsamples UMNSRS_similarity =================\n")
    outputfile.write("\n========================= Use CBOW method =================\n")            
    # run Pearson and spearman rank correlations with human ratings
    pearson, spearman, oovr = MTSmodel_CBOW.wv.evaluate_word_pairs('C:/Users/49628/Desktop/UMN/2020 Spring/BioNLP/HW2/UMNSRS_similarity.tsv')
    outputfile.write("Perason R: "+str(pearson[0])+"\n")
    outputfile.write("Spearman Rho: " + str(spearman[0])+"\n")
    outputfile.write("OOV rate: " + str(oovr))
    outputfile.write("\n====================================================================\n")
    outputfile.write("\n========================= Use SG method =================\n")        
    pearson, spearman, oovr = MTSmodel_SG.wv.evaluate_word_pairs('C:/Users/49628/Desktop/UMN/2020 Spring/BioNLP/HW2/UMNSRS_similarity.tsv')
    outputfile.write("Perason R: "+str(pearson[0])+"\n")
    outputfile.write("Spearman Rho: " + str(spearman[0])+"\n")
    outputfile.write("OOV rate: " + str(oovr))
    outputfile.write("\n====================================================================\n")
         
     # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    # ---------------------------------------------------------------- #
    # ADD YOUR CODE for evaluation on UMNSRS_relatedenss HERE ##############
    # ---------------------------------------------------------------- #   
    outputfile.write("\n========================= MTsamples UMNSRS_relatedness =================\n")
    outputfile.write("\n========================= Use CBOW method =================\n")            
    # run Pearson and spearman rank correlations with human ratings
    pearson, spearman, oovr = MTSmodel_CBOW.wv.evaluate_word_pairs('C:/Users/49628/Desktop/UMN/2020 Spring/BioNLP/HW2/UMNSRS_relatedness.tsv')
    outputfile.write("Perason R: "+str(pearson[0])+"\n")
    outputfile.write("Spearman Rho: " + str(spearman[0])+"\n")
    outputfile.write("OOV rate: " + str(oovr))
    outputfile.write("\n====================================================================\n")
    outputfile.write("\n========================= Use SG method =================\n")        
    pearson, spearman, oovr = MTSmodel_SG.wv.evaluate_word_pairs('C:/Users/49628/Desktop/UMN/2020 Spring/BioNLP/HW2/UMNSRS_relatedness.tsv')
    outputfile.write("Perason R: "+str(pearson[0])+"\n")
    outputfile.write("Spearman Rho: " + str(spearman[0])+"\n")
    outputfile.write("OOV rate: " + str(oovr))
    outputfile.write("\n====================================================================\n")
   


    # adapt an existing model to a new domain
    # first, update the vocabulary of the existing model
    # to include new words from the new domain
    # second, continue training the model
    # use CBOW method
    cord19model_CBOW.build_vocab(sentences, update=True)
    cord19model_CBOW.train(sentences, total_examples=cord19model_CBOW.corpus_count, epochs=cord19model_CBOW.iter)
    # use SG method
    cord19model_SG.build_vocab(sentences, update=True)
    cord19model_SG.train(sentences, total_examples=cord19model_SG.corpus_count, epochs=cord19model_SG.iter)

    outputfile.write("\n========================= Adapted CORD19 UMNSRS_similarity =================\n")
    outputfile.write("\n========================= Use CBOW method =================\n")            
    # run Pearson and spearman rank correlations with human ratings
    pearson, spearman, oovr = cord19model_CBOW.wv.evaluate_word_pairs('C:/Users/49628/Desktop/UMN/2020 Spring/BioNLP/HW2/UMNSRS_similarity.tsv')
    outputfile.write("Perason R: "+str(pearson[0])+"\n")
    outputfile.write("Spearman Rho: " + str(spearman[0])+"\n")
    outputfile.write("OOV rate: " + str(oovr))
    outputfile.write("\n====================================================================\n")
    outputfile.write("\n========================= Use SG method =================\n")            
    # run Pearson and spearman rank correlations with human ratings
    pearson, spearman, oovr = cord19model_SG.wv.evaluate_word_pairs('C:/Users/49628/Desktop/UMN/2020 Spring/BioNLP/HW2/UMNSRS_similarity.tsv')
    outputfile.write("Perason R: "+str(pearson[0])+"\n")
    outputfile.write("Spearman Rho: " + str(spearman[0])+"\n")
    outputfile.write("OOV rate: " + str(oovr))
    outputfile.write("\n====================================================================\n")

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    # ---------------------------------------------------------------- #
    # ADD YOUR CODE for evaluation on UMNSRS_relatedness HERE ##############
    # ---------------------------------------------------------------- #
    outputfile.write("\n========================= Adapted CORD19 UMNSRS_relatedness =================\n")
    outputfile.write("\n========================= Use CBOW method =================\n")            
    # run Pearson and spearman rank correlations with human ratings
    pearson, spearman, oovr = cord19model_CBOW.wv.evaluate_word_pairs('C:/Users/49628/Desktop/UMN/2020 Spring/BioNLP/HW2/UMNSRS_relatedness.tsv')
    outputfile.write("Perason R: "+str(pearson[0])+"\n")
    outputfile.write("Spearman Rho: " + str(spearman[0])+"\n")
    outputfile.write("OOV rate: " + str(oovr))
    outputfile.write("\n====================================================================\n")
    outputfile.write("\n========================= Use SG method =================\n")            
    # run Pearson and spearman rank correlations with human ratings
    pearson, spearman, oovr = cord19model_SG.wv.evaluate_word_pairs('C:/Users/49628/Desktop/UMN/2020 Spring/BioNLP/HW2/UMNSRS_relatedness.tsv')
    outputfile.write("Perason R: "+str(pearson[0])+"\n")
    outputfile.write("Spearman Rho: " + str(spearman[0])+"\n")
    outputfile.write("OOV rate: " + str(oovr))
    outputfile.write("\n====================================================================\n")

    ############################################
    ############################################
    # Google News SG model
    ###########################################
    ############################################

    # need this to download the model - only once (√)
    gnmodel = api.load('word2vec-google-news-300')



    # Example to show how to get the 5 most similar words to "car" or "minivan"
    print(gnmodel.most_similar(positive=['sars', 'coronavirus'], topn=5))

    outputfile.write("\n========================= Google news UMNSRS_similarity =================\n")
    # run Pearson and spearman rank correlations with human ratings
    pearson, spearman, oovr = gnmodel.evaluate_word_pairs('C:/Users/49628/Desktop/UMN/2020 Spring/BioNLP/HW2/UMNSRS_similarity.tsv')
    outputfile.write("Perason R: "+str(pearson[0])+"\n")
    outputfile.write("Spearman Rho: " + str(spearman[0])+"\n")
    outputfile.write("OOV rate: " + str(oovr))
    outputfile.write("\n====================================================================\n")

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    # ---------------------------------------------------------------- #
    # ADD YOUR CODE for evaluation on UMNSRS_relatedness HERE ##########
    # ---------------------------------------------------------------- #
    outputfile.write("\n========================= Google news UMNSRS_relatedness =================\n")
    # run Pearson and spearman rank correlations with human ratings
    pearson, spearman, oovr = gnmodel.evaluate_word_pairs('C:/Users/49628/Desktop/UMN/2020 Spring/BioNLP/HW2/UMNSRS_relatedness.tsv')
    outputfile.write("Perason R: "+str(pearson[0])+"\n")
    outputfile.write("Spearman Rho: " + str(spearman[0])+"\n")
    outputfile.write("OOV rate: " + str(oovr))
    outputfile.write("\n====================================================================\n")
    
    
    outputfile.close()


    ###############################################
    ###############################################
    # Working with embeddings
    ###############################################
    ###############################################

    # A common operation is to retrieve the vocabulary of a model.
    for i, word in enumerate(gnmodel.vocab):
        if i == 10:
            break
        print(word)


    # We can easily obtain vectors for terms the model is familiar with:
    vec_sars = gnmodel['sars']

    # Unfortunately, the model is unable to infer vectors for unfamiliar words.
    try:
        vec_cameroon = gnmodel['cameroon']
    except KeyError:
        print("The word 'cameroon' does not appear in this model")

    # examples showing how to compute similarity between pairs of words

    pairs = [
        ('car', 'minivan'),   # a minivan is a kind of car
        ('car', 'bicycle'),   # still a wheeled vehicle
        ('car', 'airplane'),  # ok, no wheels, but still a vehicle
        ('car', 'cereal'),    # ... and so on
        ('car', 'communism'),
    ]
    for w1, w2 in pairs:
        print('%r\t%r\t%.2f' % (w1, w2, gnmodel.similarity(w1, w2)))

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    # ------------------------------------------------------------------------------ #
    # ADD YOUR CODE to calculate the degree to which the following terms are related
    # using the Google News model
    # "aspirin" vs. "headache"
    # "aspirin" vs. "heart"
    # "aspirin" vs. "counter"
    # "aspirin" vs. "cat"
    # ------------------------------------------------------------------------------ #
    pairs = [
        ('aspirin', 'headache'),
        ('aspirin', 'heart'),
        ('aspirin', 'cardiac'),
        ('aspirin', 'counter'),
        ('aspirin', 'cat'),
    ]
    for w1, w2 in pairs:
        print('%r\t%r\t%.2f' % (w1, w2, gnmodel.similarity(w1, w2)))
    for w1, w2 in pairs:
        print('%r\t%r\t%.2f' % (w1, w2, cord19model_SG.similarity(w1, w2)))
    for w1, w2 in pairs:
        print('%r\t%r\t%.2f' % (w1, w2, MTSmodel_SG.similarity(w1, w2)))
        
        
# extra credits: visualize the word2vec models 
    # plot of MTSmodel_SG        
    display_pca_scatterplot(MTSmodel_SG, 
                        ['aspirin', 'headache','heart','cardiac','counter','cat'])  
    # plot of adapted cord19model
    display_pca_scatterplot(cord19model_SG, 
                        ['aspirin', 'headache','heart','cardiac','counter','cat'])
    # plot of gnmodel
    display_pca_scatterplot(gnmodel, 
                        ['aspirin', 'headache','heart','cardiac','counter','cat'])
        
        