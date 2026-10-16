from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from scipy.linalg import eigh
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).parent / 'results'
ROOT.mkdir(exist_ok=True)
RNG = np.random.default_rng(5644)

def log_gauss(X, mean, cov):
    return multivariate_normal.logpdf(X, mean=mean, cov=cov)

def confusion_conditional(y, predictions, classes):
    counts = confusion_matrix(y, predictions, labels=classes).T
    probabilities = counts / counts.sum(axis=0, keepdims=True)
    return counts, probabilities

def binary_roc(scores, labels, priors):
    # Rank observations from most likely to be class 1 to least likely.
    order = np.argsort(scores)[::-1]
    ranked_scores = scores[order]
    ranked_labels = labels[order]
    last = np.r_[np.where(np.diff(ranked_scores) != 0)[0], len(scores)-1]
    true_positive = np.cumsum(ranked_labels == 1)[last]
    false_positive = np.cumsum(ranked_labels == 0)[last]
    tpr = np.r_[0, true_positive / sum(labels == 1)]
    fpr = np.r_[0, false_positive / sum(labels == 0)]
    errors = priors[0] * fpr + priors[1] * (1-tpr)
    best = np.argmin(errors)
    thresholds = np.r_[np.inf, ranked_scores[last]]
    return fpr, tpr, errors, best, thresholds

def question1():
    mu0=np.full(4,-.5);mu1=np.ones(4)
    C0=np.array([[2,-.5,.3,0],[-.5,1,-.5,0],[.3,-.5,1,0],[0,0,0,2]])/4
    C1=np.array([[1,.3,-.2,0],[.3,2,.3,0],[-.2,.3,1,0],[0,0,0,3]])
    priors=np.array([.35,.65]); y=RNG.choice(2,size=10000,p=priors)
    X=np.empty((len(y),4));X[y==0]=RNG.multivariate_normal(mu0,C0,(y==0).sum());X[y==1]=RNG.multivariate_normal(mu1,C1,(y==1).sum())
    np.savez(ROOT/'q1_samples.npz',X=X,y=y)
    scores={"True Gaussian":log_gauss(X,mu1,C1)-log_gauss(X,mu0,C0),
            "Naive Bayes":log_gauss(X,mu1,np.diag(np.diag(C1)))-log_gauss(X,mu0,np.diag(np.diag(C0)))}
    M=np.stack([X[y==j].mean(axis=0) for j in [0,1]])
    Cs=[np.cov(X[y==j],rowvar=False,bias=True) for j in [0,1]]
    Sw=Cs[0]+Cs[1]; diff=M[1]-M[0]; Sb=np.outer(diff,diff)
    vals,vecs=eigh(Sb,Sw);w=vecs[:,-1]
    if w@diff<0:w=-w
    scores['Fisher LDA']=X@w
    print('\nQUESTION 1: prior counts',np.bincount(y),'LDA w',w, 'LDA means',M)
    fig,ax=plt.subplots(figsize=(7.5,5.8))
    results={}
    for name,ss in scores.items():
        fpr,tpr,errs,ix,thresholds=binary_roc(ss,y,priors)
        ax.plot(fpr,tpr,label=name,linewidth=1.5)
        ax.scatter(fpr[ix],tpr[ix],s=80,marker='X',zorder=4)
        th=thresholds[ix]
        theo=np.log(priors[0]/priors[1]) if name!='Fisher LDA' else np.nan
        # errors for theoretical threshold with class pdf (or NB)
        ptheo=np.mean((ss>theo)!=y) if name!='Fisher LDA' else np.nan
        result={'emp_threshold':float(np.exp(th)) if name!='Fisher LDA' else float(th),
             'log_emp_threshold':float(th),'theoretical_gamma':float(np.exp(theo)) if name!='Fisher LDA' else None,
             'emp_weighted_error':float(errs[ix]),'count_error_at_best':int(np.sum((ss>th)!=y)),
             'fpr':float(fpr[ix]),'tpr':float(tpr[ix]),'theoretical_threshold_unweighted_count_error':float(ptheo) if name!='Fisher LDA' else None}
        results[name]=result
        print(name,result)
    ax.set(xlabel='False positive rate P(D=1|L=0)',ylabel='True positive rate P(D=1|L=1)',title='Question 1: ROC, X = empirical best operating point',xlim=(-.01,1.01),ylim=(-.01,1.01))
    ax.grid(alpha=.3);ax.legend();fig.tight_layout();fig.savefig(ROOT/'q1_roc.png',dpi=175);plt.close(fig)
    return results

def question2():
    mus=np.array([[0.,0,0],[2.6,0,0],[1.3,2.25,0],[1.3,.75,2.15]])
    Cs=np.repeat(np.eye(3)[None,:,:],4,axis=0)
    priors=np.array([.3,.3,.4]);N=10000; y=RNG.choice([1,2,3],size=N,p=priors)
    comp=np.where(y==1,0,np.where(y==2,1,RNG.choice([2,3],size=N)))
    X=np.empty((N,3))
    for j in range(4):X[comp==j]=RNG.multivariate_normal(mus[j],Cs[j],(comp==j).sum())
    np.savez(ROOT/'q2_samples.npz',X=X,y=y,component=comp,mus=mus)
    lp=np.stack([log_gauss(X,mus[j],Cs[j]) for j in range(4)],axis=1)
    joint=np.column_stack([lp[:,0]+np.log(.3),lp[:,1]+np.log(.3),logsumexp(lp[:,2:4],axis=1)+np.log(.2)])
    posterior=np.exp(joint-logsumexp(joint,axis=1,keepdims=True))
    results={}
    for factor in (1,10,100):
        lam=np.array([[0,1,factor],[1,0,factor],[1,1,0]])
        risks=posterior@lam.T
        pred=np.argmin(risks,axis=1)+1
        ct,cm=confusion_conditional(y,pred,[1,2,3]); empirical=np.mean(lam[pred-1,y-1]);err=np.mean(pred!=y)
        print('\nQUESTION 2 loss factor',factor,'counts (rows decisions)',ct,'\nP(D=i|L=j)',cm,'\nerror',err,'mean risk',empirical)
        results[factor]={'counts':ct.tolist(),'conditional_confusion':cm.tolist(),'error':float(err),'empirical_risk':float(empirical)}
        fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(111,projection='3d')
        for cl,marker in [(1,'o'),(2,'^'),(3,'s')]:
            for good,color in [(True,'green'),(False,'red')]:
                mask=(y==cl)&((pred==y)==good)
                ax.scatter(*X[mask].T,c=color,marker=marker,s=5,alpha=.25 if good else .7,rasterized=True)
        ax.set(xlabel='x1',ylabel='x2',zlabel='x3',title=f'Q2: loss factor {factor}; colors green=correct / red=wrong')
        fig.tight_layout();fig.savefig(ROOT/f'q2_scatter_{factor}.png',dpi=155);plt.close(fig)
    return results


if __name__ == '__main__':
    question1()
    question2()
