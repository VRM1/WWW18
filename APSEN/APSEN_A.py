'''
the proposed aspect-specific sentiment model APSEN first
version, where we assign topic for a whole sentence. (no pseudo documents included)
'''
import numpy as np
from GetWordVec_V2 import GetWordVec, getMovieDat, getDatN
from scipy.sparse import coo_matrix
from QuickTools import nrmLize
import warnings
from django.utils.encoding import smart_str
import os
import sys
import itertools
from collections import defaultdict
from datetime import datetime
import timeit
# warnings.filterwarnings('error')
# np.seterr(all='print')
class APSEN:
    
    def __init__(self,qry,fil,K,rel,itr,typ,opfldr):
        
        ''' 
        Notes:
        1.The dstruc contains the following format
          [doc1(start of sentence, end of sentence)...docn()], which indicates
          which set of sentences belong to a document. For instance, in example
           [(0,4),(4,6)] indicates in matrix "train_d" rows 0-4 correspond to
           the sentences of document 1, 4-6 for document 2 and so on.
          
        2.The row of the "train_d" matrix indicates the sentences (not the documents)
          from all the document corpora. The "dstruc" variable serves a lookup
          for sentence-document mapping.'''
        if (typ=='ABSA'):
            self.doc_len,self.sdoc_map,self.train_d,\
            self.f_nam=GetWordVec(fil)
        if (typ=='movie'):
            self.doc_len,self.sdoc_map,self.train_d \
            ,self.f_nam=getDatN(fil,qry)
        if typ=='icml-14':
            self.doc_len,self.sdoc_map,\
            self.train_d,self.f_nam=getDatN(fil,qry)
        # the user query
        self.query=set(qry)
        np.random.seed(0)
        '''testing'''
        self.cnt=0
        '''testing'''
        self.itr_cnt=itr
        # total main-document count
        self.m_doc=self.doc_len
        # total sentence-document count
        self.t_docs=self.train_d.shape[0]
        # total words
        self.t_wrd=self.train_d.shape[1]
        # total topics
        self.K=K
        # prior for \theta
        self.alpha_=0.1
        # prior for \phi^{AZ} and \phi^{SZ}
        self.psi_=0.1
        # prior for relevance states
        self.lambda_=0.1
        # prior for \phi^{G}
        delta_=0.1
        ''' the topic document count for distribution \theta
        this is a part where I need to figure out the sentence-document-topic
        and the main-document-topic matrices. n_z_d denotes the topic assignment for
        sentence.'''
        self.n_z_m=np.zeros((self.K,self.m_doc))+self.alpha_
        # the topic word count for aspect words \phi^{AZ}
        self.n_zv_a=np.zeros((self.K,self.t_wrd))+self.psi_
        # the topic word count for sentiment words \phi^{SZ}
        self.n_zv_s=np.zeros((self.K,self.t_wrd))+self.psi_
        # the general word count for \phi^{G}
        self.n_v=np.zeros((self.t_wrd,))+delta_
        # the relevance-word count for \lambda
        self.n_r_v=np.zeros((rel,self.t_wrd))+self.lambda_
        self.r=rel
        '''other necessary counts for implementation purpose'''
        # the lookup table for words
        self.lookup_i={}
        # lookup table for sentences
        self.lookup_j={}
        # count indicating the total topic assignment for each document
        self.nd = np.zeros((self.m_doc,))
        '''count indicating the total topic assignment for words across all documents.
        Note: we might need nza nzs for aspect and sentiment words separately.'''
        self.nz= np.zeros((self.K,))
        # get the positive and sentiments from opinion lexicon
        p_data,n_data=open('dataset/positive-words.txt','r')\
                      ,open('dataset/negative-words.txt','r')
        self.p_sent={line.strip() for line in p_data} 
        self.n_sent={line.strip() for line in n_data}
        '''A lookup table to keep track of the following: for each document, which 
        word was assigned to which topic and which relevancy (i.e. 0-2)'''
        for cnt,d in enumerate(self.train_d):
            d=coo_matrix(d)
            '''the bool value is used to check if the word
             in a sentence was assigned  to topic (True) or background (False)'''
            tmp=dict([(v,[0,bool]) for v in d.col])
            self.lookup_i[cnt]=tmp
            self.lookup_j[cnt]=0
        # output folder
        self.opfldr=opfldr
        # get one of the query words for saving the output
        self.q=next(iter(qry))
    
    # method to randomly initialize the parameters
    def rand_assign(self):

        for i,d in enumerate(self.train_d):
            
            # randomly assign a topic for a whole sentence (rather than every word)
            z = np.random.randint(0, self.K)
            # get the document the sentence i is assigned to
            m=self.sdoc_map[i]
            d=coo_matrix(d)
            # some sentences are without any words since all of them are a part of stop words
            if not len(d.col): 
                continue
            # update the topic-document matrix count
            self.n_z_m[z,m]+=1
            # update the total topic-document assignment count
            self.nd[m]+=1
            # update the lookup table entry for the sentence-topic
            self.lookup_j[i]=z

            for v in d.col:
                # obtain a relevance 
                r = np.random.randint(0, self.r)
                # update the counts
                self.n_r_v[r,v]+=1
                self.nz[z]+=1
                ''' the random update of the aspect-topic and aspect-specific sentiment-topic
                 counts are decided based on the relevancy r.v. r. However, this selective
                 update during random initialization is not really necessary.(I guess)'''
                if r==0:
                    self.n_v[v]+=1
                    self.lookup_i[i][v][1]=False
                if r==1:
                    self.n_zv_a[z,v]+=1
                    self.lookup_i[i][v][0]=z
                    self.lookup_i[i][v][1]=True
                    
                
                
    # method to calculate the topic posterior
    def __get_tpPost(self,d,m,V,z,typ):
        
        
        # calculate the posterior z for a sentence rather than a word
        if typ=='Aspect':
            # numpy array to hold the sampled topics
            s_tpics=np.zeros((self.K,))
            for z in range(self.K):
                p=0
                t1=1
                t2=self.n_z_m[z,m] / (self.nd[m]+self.alpha_ * self.K)
                for w in V:
                    if self.lookup_i[d][w][1] == True:
                        # get the number of times word w was assigned to relevance 2 (i.e. aspect)
                        n_r_vw=self.n_r_v[1,w]
                        # get number of times topic z is assigned word w
                        n_zv_a=self.n_zv_a[z,w]
                        # get number of times topic z is assigned across all words
                        nz=self.nz[z]
                        for j in range(int(n_r_vw)):
                            t1 = t1 * (n_zv_a + j)/(nz + self.psi_ * self.t_wrd + p)
                            p+=1
                s_tpics[z] = t1 * t2
            if np.sum(s_tpics)!=0:
                s_tpics=s_tpics/np.sum(s_tpics)
            else:
                s_tpics+=0.1
                s_tpics = s_tpics/np.sum(s_tpics)
            return s_tpics
    # method to calculate the posterior of the relevance
    def __get_relPost(self,v,new_za):
        
        ''' The following conditions override the probabilities:
        1. if the word is present in the opinion corpus it is a background word.
        2. if the word matches with the user query then it's an aspect word.
        '''
        # get the original string word
        wrd=self.f_nam[v]
        n_za=self.n_zv_a[new_za,v] / (self.nz[new_za]+self.psi_ * self.t_wrd)
        n_g=self.n_v[v] / np.sum(self.n_v)
        
        new_rg=self.n_r_v[0,v]/(np.sum(self.n_r_v[:,v])) * n_g
        new_ra=self.n_r_v[1,v]/(np.sum(self.n_r_v[:,v])) * n_za
#         if wrd in self.p_sent or wrd in self.n_sent:
#             # relevance 0 indicates it's a background word
#             new_rg=1
        if wrd in self.query:
            # if the word is in query then it must be an aspect word
            new_ra=1
        new_r=[new_rg,new_ra]
        new_r=new_r/np.sum(new_r)
        return new_r
        
    # method to perform inference
    def inference(self):
        ''' reformat training data into sentences as key and the words as values
        since iterating through the sparse matrix is extremely slow'''
        new_train=defaultdict()
        for i,d in enumerate(self.train_d):
            d=coo_matrix(d)
            new_train[i]=d.col
        
        
        for itr in xrange(self.itr_cnt):
            tmp=[]
#             print now.strftime("%Y-%m-%d%H:%M")
            start_time = timeit.default_timer()
            for i in new_train:
                
                sen_wrd = new_train[i]
                # perform inference for the sentence row
                # if sentence has no words (removed since all were top words) continue
                if not len(sen_wrd):
                    continue

                # get the document corresponding to the sentence
                m=self.sdoc_map[i]
                # get the topic assignment for the sentence
                z=self.lookup_j[i]
                # remove the assignment of topic for document m
                self.n_z_m[z,m]-=1
                # decrement the total topic assignment for the document
                self.nd[m]-=1 
                for v in sen_wrd:
                    if self.lookup_i[i][v][1] == True:
                        # remove the topic assignment for all the words
                        self.n_zv_a[z,v]-=1
                        # decrement global topic assignment
                        self.nz[z]-=1

                '''sample new topics for aspect, aspect-specific
                 sentiment and generic (or background) words'''
                new_za=self.__get_tpPost(i,m,sen_wrd,z,'Aspect')
               
                new_za=np.random.multinomial(1,new_za).argmax()
                # update document topic count
                self.n_z_m[new_za,m]+=1
                # update the entry for the newly sampled aspect topic in the lookup table
                self.lookup_j[i]=new_za
                self.nd[m]+=1
                for v in sen_wrd:
                    r=self.lookup_i[i][v][1]
                    # remove randomly relevancy count
                    if r==False:
                        self.n_r_v[0,v]-=1
                    else:
                        self.n_r_v[1,v]-=1
                        
                    # sample a new relevancy
                    new_r=self.__get_relPost(v,new_za)
                    # pick the new relevance
                    new_r=np.random.multinomial(1,new_r).argmax()
                    # update the counts
                    self.n_r_v[new_r,v]+=1
                        
                    if new_r==0:
                        # update the background word distribution
                        self.n_v[v]+=1
                        self.lookup_i[i][v][1]=False
                    if new_r==1:
                        self.n_zv_a[new_za,v]+=1
                        self.nz[new_za]+=1
                        # update the newly sampled aspect topic in the lookup table
                        self.lookup_i[i][v][0]=new_za
                        self.lookup_i[i][v][1]=True
                
            elapsed = timeit.default_timer() - start_time
            print elapsed

        print 'writing lambda, phi^{AZ}, phi^{SZ} and \phi^{G} parameters to as outputs'
        self._writ_oput()
        
                    
                        
    def _writ_oput(self):
        
#         path='output/'+self.opfldr+'/'+self.q+'/'
        path='output/'
        if not os.path.exists(path):
            os.makedirs(path)
        # save the feature names
        with open(path+'feature_nam.mat','w') as fp:
            np.save(fp,np.array(self.f_nam))
        ''' normalize \phi^{SZ} p(senti word | z) self.n_zv_s 
        row normalize in other words take sum across all colums for every z'''
        self.n_zv_a=nrmLize(self.n_zv_a)
        self.n_v=nrmLize(self.n_v)
        # check the relevance p(w|r) from the distribution \lambda
        self.n_r_v=nrmLize(self.n_r_v)
        with open(path+'lambda_.mat','w') as fp:
            np.save(fp,self.n_r_v)
        with open(path+'phi_za.mat','w') as fp:
            np.save(fp,self.n_zv_a)
        with open(path+'phi_g.mat','w') as fp:
            np.save(fp,self.n_v)
    
def displayTopics(top_k,typ,qry,nam):
    
    # get a random query word for saving
    q=next(iter(qry))
#     path='output/'+nam+'/'+q+'/'
    path='output/'
    f_nam=np.load(path+'feature_nam.mat')
    lambda_=np.load(path+'lambda_.mat')
    if typ=='Aspects':
        phi=np.load(path+'phi_za.mat')
    if typ=='Sentiments':
        phi=np.load(path+'phi_zs.mat')
    if typ=='Background':
        phi=np.load(path+'phi_g.mat')
    if typ=='Aspects' or typ=='Sentiments':
        for i,topics in enumerate(phi):
            tp_wrds = np.array(f_nam)[np.argsort(topics)[-top_k:][::-1]]
            tmp=set(tp_wrds)
            if len(qry.intersection(tmp))==len(qry):
                print('Topic {}: {}'.format(i,smart_str(' '.join(tp_wrds))))
    if typ=='Background':
        bk_wrds = np.array(f_nam)[np.argsort(phi)[-top_k:][::-1]]
        for w in bk_wrds:print w
    if typ=='Relevance':
        for i,rel in enumerate(lambda_):
            rel_wrds = set(np.array(f_nam)[np.argsort(rel)[-top_k:][::-1]])
            print('Relevance {}: {}'.format(i,' '.join(rel_wrds)))
                
            
if __name__ == '__main__':
    
    
    d_sets={'icml-14':'dataset/home_theatre.pickle',\
            'ABSA':'dataset/ABSA-15_Restaurants_Train_Final.xml',\
     'movie-1':'dataset/desolation_smaug.pickle', 'movie-2':'dataset/cptain_civil_war.pickle'}
    #location of input data global settings
#     qry=['screen','brightness','quality','display','lcd','reflectiveness']
    qry=set(['smaug'])
    mov_nam='movie-1'
    absa=APSEN(qry,d_sets[mov_nam],K=50,rel=2,itr=250,typ='movie',opfldr=mov_nam)
    absa.rand_assign()
    absa.inference()

    displayTopics(11, 'Aspects',qry,mov_nam)