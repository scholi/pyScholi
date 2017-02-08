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
