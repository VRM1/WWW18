# this package provides visualization of the APSEN results
import numpy as np
import matplotlib.pyplot as plt
# module to view the historgram of topic posterior theta
def viewDocTpDtr():
    
    theta_=np.load('output/theta_.mat')
    D = theta_.shape[1]
    tp_h=[np.count_nonzero(theta_[:,d]) for d in xrange(D)]
    tp_d=[]
    for d in xrange(D):
        for t in zip(*theta_[:,d].nonzero()):
            tp_d.append(t[0])
#     tp_d=[t[0] for t in zip(*theta_[:,d].nonzero()) for d in xrange(D)]
    plt.hist(tp_d,50)
    plt.show()
        
def viewDistr(mat,argg):
    
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    arr=np.sum(mat,axis=0) if argg=='row' else np.sum(mat,axis=1)
    plt.xticks(np.arange(len(arr)))
#     line,=ax.plot(np.arange(len(arr)),arr,marker='o',alpha=0.8)
    bar_g=ax.bar(np.arange(len(arr)),arr,align='center',alpha=0.5, width=0.1)
#     ax.plot(np.arange(len(arr)),arr,marker='o',linestyle="",alpha=0.8)
    plt.show()
    return fig,ax

if __name__ == '__main__':
    
    chk=np.random.randint(10,size=(10,15))
    viewDistr(chk,argg='row')
    
    