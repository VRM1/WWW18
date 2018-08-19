'''
the proposed aspect-specific sentiment model APSUM
Variation 2 without the topic aggregator.[i.e, Figure 2(b) of our paper
"A Sparse Topic Model for Extracting Aspect-Specific
Summaries from Online Reviews" without aggregator]
'''
import numpy as np
from GetWordVec_V2 import GetWordVec, getMovieDat, getDatN
from scipy.sparse import lil_matrix
from QuickTools import nrmLize
from django.utils.encoding import smart_str
import timeit
from itertools import izip,product
from collections import defaultdict
from scipy.special import gammaln
# from GetStats import viewDocTpDtr
# np.random.seed(0)
class APSEN:
    
    def __init__(self,qry,fil,K,itr,typ):
        
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
        '''testing'''
        self.cnt=0
        # the number of relevance states (fixed)
        self.r=2
        '''testing'''
        self.itr_cnt=itr
        # total document count
        self.t_docs=self.train_d.shape[0]
        # total words
        self.t_wrd=self.train_d.shape[1]
        # total topics
        self.K=K
        # prior for \theta
        self.alpha_=0.1
        # prior for \phi^{AZ} and \phi^{SZ}
        self.beta_=0.01
        # prior for relevance states
        self.lambda_=0.1
        # prior for \phi^{B}
        delta_=0.001
        # the topic document count for distribution \theta
        self.n_z_d=np.zeros((self.K,self.t_docs))
        # the topic word count for aspect words \phi^{AZ}
        self.n_zv_a=np.zeros((self.K,self.t_wrd))
        # the topic word count for sentiment words \phi^{SZ}
        self.n_zv_s=np.zeros((self.K,self.t_wrd))
        # the general word count for \phi^{G}
        self.n_v=np.zeros((self.t_wrd,))+delta_
        # the relevance-word count for \lambda
        self.n_r_v=np.zeros((self.r,self.t_wrd))+self.lambda_
        '''other necessary counts for implementation purpose'''
        # the lookup table
        self.lookup_i=defaultdict(dict)
        # count indicating the total topic assignment for each document
        self.nd = np.zeros((self.t_docs,))
        '''count indicating the total topic assignment for words across all documents.
        Note: we might need nza nzs for aspect and sentiment words separately.'''
        self.nz= np.zeros((self.K,))
        # get the positive and negative sentiments from opinion lexicon
        p_data,n_data=open('dataset/positive-words.txt','r')\
                      ,open('dataset/negative-words.txt','r')
        self.p_sent={line.strip() for line in p_data} 
        self.n_sent={line.strip() for line in n_data}
        # using tocoo() method of sparse matrix makes the iteration over sparse matrix extremely fast
        self.td = self.train_d.tocoo()
        '''A lookup table to keep track of the following: for each document, which 
        word was assigned to which topic and which relevancy (i.e. 0-1)'''
        
        for i,v in izip(self.td.row, self.td.col):
            self.lookup_i[i][v]=[0,0]
        ''' The following are the count variables for
         the spike and slab part. For now, we are ignoring the aggregator
         document part. Instead, we assume every document d is a aggregator document so d=l'''
        # prior epsion eta and eta`
        self.eta_=0.1
        # weak smoothing prior
        self.alphaHat_=0.00001
        # times a doc was assigned to RV c=0 or c=1 (count for \pi distr)
        self.n_l_c=np.zeros((self.t_docs,2))+self.eta_
        ''' another count for bernoulli variable c, 
        a K*L matrix This matrix is different from the document-topic matrix, where the
        values as simply binary, which indicates whether a topic was assigned to a document
        or not. THIS IS NOT A COUNT MATRIX. [Remark:
        we can use sparse matrix here since the values are 1 or 0]'''
        self.n_z_cl=np.zeros((self.K,self.t_docs))
        # start random assignment
        self.rand_assign()
    # method to randomly initialize the parameters
    def rand_assign(self):
        
        '''1. First randomly sample for the spike and slab variable
        2. Now when randomly drawing a topic z for a word, use
        this sub-space of topic dimension from the variable 
        self.n_z_cl and sample only those topics for which the topic
        z==1'''
        for d in xrange(self.t_docs): 
            # choose a r.v c 
            choic=[0]
            # sometimes the choice variable generates all zeros
            while sum(choic) == 0:
                choic=np.random.choice(2,self.K,p=[0.5,0.5])
            for z,c in enumerate(choic):
                self.n_l_c[d][c]+=1
                if c==1:
                    self.n_z_cl[z,d]=1
        
        for i,v in izip(self.td.row,self.td.col):
            # get the indices of topics from the sub-space of the topic
            I_c=[t[0] for t in zip(*self.n_z_cl[:,i].nonzero())]
            # randomly pick a topic and a relevance
            z = np.random.choice(I_c,1)[0]
            r = np.random.randint(0, self.r)
            self.lookup_i[i][v][1]=r
            self.n_r_v[r,v]+=1
            
            ''' the random update of the aspect-topic and aspect-specific sentiment-topic
             counts are decided based on the relevancy r.v. r. However, this selective
             update during random initialization is not really necessary.(I guess)'''
            if r==0:self.n_v[v]+=1
            if r==1:
                self.n_zv_a[z,v]+=1
                # update the counts
                self.n_z_d[z,i]+=1
                self.nz[z]+=1
                self.nd[i]+=1
                self.lookup_i[i][v][0]=z
            if r==2:
                self.n_zv_s[z,v]+=1
                # update the counts
                self.n_z_d[z,i]+=1
                self.nz[z]+=1
                self.nd[i]+=1
                self.lookup_i[i][v][0]=z
            
        
        
    # method to calculate the topic posterior
    def __get_tpPost(self,d,v,typ):
        
        
        t_1 = self.n_z_d[:,d] + self.n_z_cl[:,d]*self.alpha_ + self.alphaHat_
        if typ=='Aspect':
            t_2 = (self.n_zv_a[:,v] + self.beta_) / (self.nz+self.beta_ * self.t_wrd)
        if typ=='Sentiment':
            t_2= (self.n_zv_s[:,v] + self.beta_) / (self.nz+self.beta_ * self.t_wrd)
        new_z=t_1*t_2
        new_z=new_z/np.sum(new_z)
        return new_z
    
    
    # method to calculate the posteriro of the relevance
    def __get_relPost(self,v,new_za):
        
        ''' The following conditions override the probabilities:
        1. if the word is present in the opinion corpus it is a background word.
        2. if the word matches with the user query then it's an aspect word.
        '''
        # get the original string word
        wrd=self.f_nam[v]
        n_za= (self.n_zv_a[new_za,v] +self.beta_) / (self.nz[new_za]+self.beta_ * self.t_wrd)
        n_g=self.n_v[v] / np.sum(self.n_v)
        
        new_rg=self.n_r_v[0,v]/(np.sum(self.n_r_v[:,v])) * n_g
        new_ra=self.n_r_v[1,v]/(np.sum(self.n_r_v[:,v])) * n_za
        if wrd in self.p_sent or wrd in self.n_sent:
            # relevance 0 indicates it's a background word
            new_rg=1
        if wrd in self.query:
            # if the word is in query then it must be an aspect word
            new_ra=1
        new_r=[new_rg,new_ra]
        new_r=new_r/np.sum(new_r)
        return new_r
    
    def __getSpikeNSlab2(self,l,z):
        
        # get the set of non-zero entries for the aggregator-document l and topics K (the identity vector)
        I_c=[t[0] for t in zip(*self.n_z_cl[:,l].nonzero())]
        # zS implies we sum over all z, equivalent of n_z*_l
        n_zS_l=np.sum(self.n_z_d[:,l])
        A_l=len(I_c)
        # get the distribution pi by normalizing count n_l_c
        pi_a=self.n_l_c[l]/np.float(np.sum(self.n_l_c[l]))
        if (self.n_z_cl[z,l] and self.n_z_d[z,l]==0):
            log_diff = gammaln(A_l*self.alpha_ + self.K*self.alphaHat_) - \
                        gammaln((A_l-1)*self.alpha_ + self.K*self.alphaHat_)
            
            log_diff -= gammaln(n_zS_l + A_l*self.alpha_ + self.K*self.alphaHat_) - \
                        gammaln(n_zS_l + (A_l-1)*self.alpha_ + self.K*self.alphaHat_)
                        
            ratio = np.exp(log_diff) * pi_a[1]/pi_a[0]
            p = ratio/float(1+ratio)
            if (np.random.rand() > p):
                self.n_z_cl[z][l]=0
                self.n_l_c[l][0] += 1
        elif(not self.n_z_cl[z,l]):
            
            log_diff = gammaln((A_l+1)*self.alpha_ + self.K*self.alphaHat_) - \
                        gammaln(A_l*self.alpha_ + self.K*self.alphaHat_)
            log_diff -= gammaln(n_zS_l + (A_l+1)*self.alpha_ + self.K*self.alphaHat_) -  \
                        gammaln(n_zS_l + A_l*self.alpha_ + self.K*self.alphaHat_)
            
            ratio = np.exp(log_diff) * pi_a[1]/pi_a[0]
            p = ratio/float(1+ratio)
            if (np.random.rand() < p):
                self.n_z_cl[z][l]=1
                self.n_l_c[l][1] += 1
                
    
    
    
    def __getSpikeNSlab(self,d,z):
        
        val_a=self.n_l_c[d][0]
        val_b=self.n_l_c[d][1]
        # get the set of non-zero entries for the document i and topics K (the identity vector)
        I_c=[t[0] for t in zip(*self.n_z_cl[:,d].nonzero())]
        A_l=len(I_c)
        n_z_l=self.n_z_d[z][d]
        deno=1
        num=1
        for j in xrange(int(n_z_l)+1):
            num *= (self.alpha_ * self.n_z_cl[z,d] + self.alphaHat_ + j)
        for j in xrange(int(self.nd[d])-1):
            deno *= (A_l*self.alpha_ + self.K*self.alphaHat_+j)
        val_c=num/deno
        p_c0=val_a * val_c
        p_c1=val_b * val_c
        new_c=[p_c0,p_c1]/np.sum([p_c0,p_c1])
        return new_c
        
    # method to perform inference
    def inference(self):
        
        for itr in xrange(self.itr_cnt):
            # first we sample for the spike and slab
            strt=timeit.default_timer()
            for d in xrange(self.t_docs):
                for z in xrange(self.K):
                    if self.n_z_cl[z,d]:
                        # decrement the randomly assigned count for topic z
                        self.n_l_c[d][1]-=1
                    else:
                        self.n_l_c[d][0]-=1
                        
                    new_c=self.__getSpikeNSlab2(d,z)
#                     # sample a new c for a z
#                     new_c=np.random.multinomial(1,new_c).argmax()
#                     self.n_l_c[d][new_c]+=1
#                     # update the indicator document/aggregator-doc- topic distribution
#                     self.n_z_cl[z,d]=new_c
                
                match=0
                ''' get the set of all words for the document d. 
                If the document (or short sentence) has a query word match. The document is
                probably relevant to the query.'''
                W=self.lookup_i[d]
                wrds=set([self.f_nam[v] for v in W])
                for q in self.query:
                    if q in wrds: match=1; break
                          
                # second, we sample for word-topic
                for v in W:
                    # get the randomly assigned topic and relevancy assignment
                    z=self.lookup_i[d][v][0]
                    r=self.lookup_i[d][v][1]
                    # remove the corresponding random topic count
                    if r==0:
                        self.n_v[v]-=1
                    if r==1:
                        self.n_z_d[z,d]-=1
                        self.n_zv_a[z,v]-=1
                        # decrement the total topic assignment for the document
                        self.nd[d]-=1
                        # decrement global topic assignment
                        self.nz[z]-=1
                        self.lookup_i[d][v][1]=0
                    '''sample new topics for aspect, aspect-specific
                     sentiment and generic (or background) words assuming we know the RV r'''
                    new_za=self.__get_tpPost(d,v,'Aspect')
                    # sample a single topic from their multinomials
                    new_za=np.random.multinomial(1,new_za).argmax()
                     
                    # remove randomly assigned topic & relevancy count
                    self.n_r_v[r,v]-=1
                    # sample a new relevancy
                    new_r=self.__get_relPost(v,new_za)
                    # pick the new relevance
                    ''' if the query word is found in the sentence, don't
                    sample a relevance, instead mark all words in this
                    sentence as relevant'''
                    if not match: 
                        new_r=np.random.multinomial(1,new_r).argmax()
                    else:
                        new_r=1
                     
                    # update the counts
                    self.n_r_v[new_r,v]+=1
                    # update the newly sampled relevance in the lookup table
                    self.lookup_i[d][v][1]=new_r
                    if new_r==0:
                        self.n_v[v]+=1
                    if new_r==1:
                        self.n_z_d[new_za,d]+=1
                        self.n_zv_a[new_za,v]+=1
                        self.nz[new_za]+=1
                        self.nd[d]+=1
                        # update the newly sampled aspect topic in the lookup table
                        self.lookup_i[d][v][0]=new_za
                     
            
            elapsed=timeit.default_timer()-strt
            print('iteration:{} done at:{} sec'.format(itr,int(elapsed)))
        print 'writing lambda, phi^{AZ}, phi^{SZ} and \phi^{G} parameters to as outputs'
        self._writ_oput()
                        
    def _writ_oput(self):
        
        # save the feature names
        with open('output/feature_nam.mat','w') as fp:
            np.save(fp,np.array(self.f_nam))
        ''' normalize \phi^{SZ} p(senti word | z) self.n_zv_s 
        row normalize in other words take sum across all colums for every z'''
        self.n_zv_a=nrmLize(self.n_zv_a)
#         self.n_zv_s=nrmLize(self.n_zv_s)
        self.n_v=nrmLize(self.n_v)
        # check the relevance p(w|r) from the distribution \lambda
        self.n_r_v=nrmLize(self.n_r_v)
        with open('output/theta_.mat','w') as fp:
            np.save(fp,self.n_z_d)
        with open('output/lambda_.mat','w') as fp:
            np.save(fp,self.n_r_v)
        with open('output/phi_za.mat','w') as fp:
            np.save(fp,self.n_zv_a)
        with open('output/phi_zs.mat','w') as fp:
            np.save(fp,self.n_zv_s)
        with open('output/phi_g.mat','w') as fp:
            np.save(fp,self.n_v)

def displayTopics(top_k,typ,qry):
    
    f_nam=np.load('output/feature_nam.mat')
    lambda_=np.load('output/lambda_.mat')
    if typ=='Aspects':
        phi=np.load('output/phi_za.mat')
    if typ=='Sentiments':
        phi=np.load('output/phi_zs.mat')
    if typ=='Background':
        phi=np.load('output/phi_g.mat')
    if typ=='Aspects' or typ=='Sentiments':
        for i,topics in enumerate(phi):
            tp_wrds = np.array(f_nam)[np.argsort(topics)[-top_k:][::-1]]
            tmp=set(tp_wrds)
            if qry.intersection(tmp):
                print('Topic {}: {}'.format(i,smart_str(' '.join(tp_wrds))))
    if typ=='Background':
        bk_wrds = np.array(f_nam)[np.argsort(phi)[-top_k:][::-1]]
        for w in bk_wrds:print w
    if typ=='Relevance':
        for i,rel in enumerate(lambda_):
            rel_wrds = set(np.array(f_nam)[np.argsort(rel)[-top_k:][::-1]])
            str_val= smart_str(' '.join(rel_wrds))
            print('Relevance {}: {}'.format(i,str_val))
           
def runModel():
    
    d_sets={'icml-14':'dataset/camIcml.pickle',\
            'ABSA':'dataset/ABSA-15_Restaurants_Train_Final.xml',\
     'movie-1':'dataset/desolation_smaug.pickle', 'movie-2':'dataset/cptain_civil_war.pickle',}
    qry=set(['portable'])
    absa=APSEN(qry,d_sets['icml-14'],K=70,itr=300,typ='movie')
    absa.inference()
    displayTopics(10, 'Aspects',set(qry))

# def getStats():
#     viewDocTpDtr()
    

if __name__ == '__main__':
    
    runModel()
