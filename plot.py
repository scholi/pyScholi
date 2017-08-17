import matplotlib.pyplot as plt
import numpy as np

def stdplot(x,y,Nsig=2,ax=None,color='b',axis=1,**kargs):
	if ax==None:
		fig, ax = plt.subplots(1,1,figsize=(8,6))
	dy=np.std(y,axis=axis)
	mean = np.mean(y,axis=axis)
	ax.plot(x,mean,color=color,**kargs)
	for i in range(1,Nsig+1):
		ax.fill_between(x,mean-i*dy,mean+i*dy,color=color,alpha=.2)

def sublegend(*ax, color='white', margin=9, titles=None, fontsize=14):
    props = dict(boxstyle='round', facecolor=color, alpha=1)  
    for i,a in enumerate(np.ravel(ax)):
        if titles is False:
            a.set_title("")
        a.annotate(chr(ord('a')+i),(0, 1),xytext=(margin,-margin),fontsize=fontsize,verticalalignment='top', bbox=props, xycoords='axes fraction',textcoords='offset pixels');
        
def plotFWHM(ax, x0, G, h, fmt=".2f", unit='px', col='k', linestyle='-', offset=(0,0), va="bottom"):
    lab = "{:"+fmt+"}"+unit
    ax.annotate("",(x0-G/2,h),(x0+G/2,h), arrowprops=dict(arrowstyle='<|-|>',color=col, linestyle=linestyle))
    ax.annotate(lab.format(G),(x0,h),offset, color=col, textcoords='offset pixels', ha="center", va=va)