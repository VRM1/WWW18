from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
import string
from AbsaParser import read_semeval2016_task5_subtask2
from imdbpie import Imdb
imdb = Imdb()
# imdb = Imdb(anonymize=True) # to proxy requests
import pickle, os, random
import nltk
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from scipy.sparse import coo_matrix
from django.utils.encoding import smart_str
random.seed(0)
'''This is a modification over previous word to vector. In this program I keep track
of the sentences of a document and the word vectors for each sentence in a document.'''
# block to create bag of words vectors
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [i for i in tokens if i not in string.punctuation]
    stems = stem_tokens(tokens, stemmer)
    return stems

# method to create a word vector for topic model
def creatWordVec(documents):
    
#     vectorizer = CountVectorizer(tokenizer=tokenize,min_df=1,stop_words='english')
    vectorizer = CountVectorizer(min_df=1,stop_words='english')
    ''' obtain the reviews as a list, where each element is 
    a single review or a document that is written by a user'''
    reviews=[]
    # a list that preserves the document structure by keeping tack of sentence and the indices of words 
    doc_struct=[]
    # a sentence-document mapper indicating which sentence belongs to which document
    sntdoc_map={}
    count=0
    for i,doc in enumerate(documents):
        doc_struct.append([count])
        for s in doc.sentences:
            sntdoc_map[count]=i
            count+=1
            reviews.append(s.text)
        doc_struct[i].append(count)
             
#     reviews=[''.join([s.text for s in doc.sentences]) \
#               for doc in documents]
    # create a vector of review words using the scikit CountVectorizer module
    rv_vec=vectorizer.fit_transform(reviews)
    # get the names of each column of the vector
    f_nam=tuple(vectorizer.get_feature_names())
    return (doc_struct,sntdoc_map,rv_vec,f_nam)

def getMovieDat(datset,max_doc):
    
    lookup={'desolation_smaug':'tt1170358'}
    vectorizer = CountVectorizer(min_df=1,stop_words='english')
    mov_id=lookup[datset]
    if os.path.isfile('dataset/'+datset+'.pickle'):
        with open('dataset/'+datset+'.pickle','r') as fp:
            documents=pickle.load(fp)
    else:
        documents=imdb.get_title_reviews(mov_id, max_results=max_doc)
        with open('dataset/'+datset+'.pickle','w') as fp:
            pickle.dump(documents,fp)
    sntdoc_map={}
    doc_struct=[]
    lmtzr = WordNetLemmatizer()
    count=0
    docs=[]
    for i,doc in enumerate(documents):
        doc_struct.append([count])
        tmp=sent_tokenize(doc.text)
#         tmp=doc.text.split('\n')
        sentences=[]
        uniq_wrd=set()
        for s in tmp:
            tokens = nltk.word_tokenize(s)
            tagged = nltk.pos_tag(tokens)
            nouns = [word for word,pos in tagged \
                     if (pos == 'NN' or pos == 'NNP' or pos == 'NNS'\
                         or pos == 'NNPS' or pos=='JJ')]
            downcased = [x.lower() for x in nouns]
            downcased = [lmtzr.lemmatize(x) for x in downcased]
            downcased = [x for x in downcased if x not in stopwords.words('english')]
            if len(downcased)>1:
                uniq_wrd.update(downcased)
                sentences.append(' '.join(downcased))
        docs.append(sentences)
        for sen in sentences:
            sntdoc_map[count]=i
            count+=1
            reviews.append(sen)
        doc_struct[i].append(count)
    rv_vec=vectorizer.fit_transform(reviews)
    # get the names of each column of the vector
    f_nam=tuple(vectorizer.get_feature_names())
    return (doc_struct,sntdoc_map,rv_vec,f_nam)    

# this method is for text reviews based on amazon products
def getDatK(fil):
    
    reviews=[]
    with open(fil,'r') as fp:
        for i,lin in enumerate(fp):
            reviews.append(lin.strip())
            if i>300: break
    vectorizer = CountVectorizer(min_df=1,stop_words='english')
    rv_vec=vectorizer.fit_transform(reviews)
    # get the names of each column of the vector
    f_nam=tuple(vectorizer.get_feature_names())
    return (rv_vec,f_nam)
    

def GetWordVec(dataset):
    fil=dataset
    dat_set=read_semeval2016_task5_subtask2(fil)
    d_struc,sdoc_map,train_d,f_nam=creatWordVec(dat_set)
    return(len(d_struc),sdoc_map,train_d,f_nam)

# this method creates the word vector documents for icml dataset with every document being a setnence
def getDatM(f_doc,f_voc):
    
    reviews=[]
    f_lukup={}
    vectorizer = CountVectorizer(min_df=1)
    doc_struct=[]
    '''
    1. In this case we donot know what sentence belong to what
    document. Therefore, every sentence is considerered a document.
    '''
    sntdoc_map={}
    count=0
    qry=''
    with open(f_voc,'r') as fp:
        for lin in fp:
            lin=lin.strip().split(':')
            if lin[1]=='picture':
                qry=lin[0]
            f_lukup[lin[0]]=lin[1]
    with open(f_doc,'r') as fp:
        for i,lin in enumerate(fp):
            doc_struct.append([count,count])
            reviews.append(lin.strip())
            sntdoc_map[count]=count
            count+=1
            if i==700: break

    rv_vec=vectorizer.fit_transform(reviews)
    f_nam=tuple(vectorizer.get_feature_names())
    f_nam=[f_lukup[k] for k in f_nam]
    return(doc_struct,sntdoc_map,rv_vec,f_nam)

# this version creates the word vector for icml dataset with every document being a collection of sentences
def getDatN(fil,qry):
    
    reviews=[]
    sntdoc_map={}
    t_sent=0
    vectorizer = CountVectorizer(min_df=1,stop_words='english')
    count=0
    with open(fil,'r') as fp:
        docs=pickle.load(fp)
#         docs=random.sample(docs,500)
        t_doc=len(docs)
        for i,d in enumerate(docs):
            sdoc=0
#             if len(qry.intersection(d[-1])) == len(qry):
            for s in d[:-1]:
                sdoc+=1
                t_sent+=1
                reviews.append(s)
                sntdoc_map[len(reviews)-1]=count
                count+=1
            # the number of sentences of my algorithm cannot be more than 1500 or else it's too slow
#             if t_sent>1300:break   
    rv_vec=vectorizer.fit_transform(reviews)
    f_nam=tuple(vectorizer.get_feature_names())
#     writTTM(fil,rv_vec,f_nam,sntdoc_map)       
    return(count,sntdoc_map,rv_vec,f_nam)

    
# method to write the data for running TTM (targeted topic model) 
def writTTM(fil,dat,feat,sntdoc_map):
    
    fp=open(fil+'.TTM','w')
    fp_j=open(fil+'.TTM.wordlist','w')
    sen_rang={}
    # collect all sentences per document
    for s in sntdoc_map:
        if sntdoc_map[s] not in sen_rang:
            sen_rang[sntdoc_map[s]]=[s]
        else:
            sen_rang[sntdoc_map[s]].append(s)
    for d in sen_rang:
        # get total sentences per-document
        n_sen=len(sen_rang[d])
        start=sen_rang[d][0]
        end=sen_rang[d][-1]
#         fp.write(str(n_sen)+'\n'+'0'+'\n')
        fp.write(str(n_sen)+'\n')
        sen_vect=dat[start:end+1]
        for sen in sen_vect:
            wrds=coo_matrix(sen)
            ''' some sentences can be empty since 
            all words are stop words in that case 
            fill a dummy value'''
            if not len(wrds.col): 
                fp.write(str(10)+'\n')
                continue
            for w in wrds.col: 
                fp.write(str(w)+' ')
            fp.write('\n')
    for f in feat:
        fp_j.write(smart_str(f)+'\n')

    
if __name__ == '__main__':
    
    dataset='desolation_smaug'
    qry='sauron'
    TmpgetDatN(dataset, qry)
