import numpy as np
def nrmLize(data):
    
    if len(data.shape) == 1:
        return(data/float(np.sum(data)))
    rws=data.shape[0]
    clm=data.shape[1]
    data=data.astype(float)
    if(clm==1):
        div=np.sum(data,axis=0)
        div=div[:,np.newaxis]
        div = np.repeat(div,rws,axis=0)
    else:
        # axis=1 across row (X axis) and axis=0 across column i.e. Y axis
        div=np.sum(data,axis=1)
        div=div[:,np.newaxis]
    data=np.divide(data,div)
    return data
# create a score for U-U self prior
def get_selfPrior(mat):
    
    for cnt,val in enumerate(mat):
        nn=val.ravel()[np.flatnonzero(val)]
        avg=np.average(nn)
        mx=np.max(nn)
        self_val=(mx+avg)/2
        mat[cnt,cnt]=self_val
    return mat