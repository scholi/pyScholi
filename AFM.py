import numpy as np
import matplotlib.pyplot as plt
import pyScholi.math as m
import scipy.optimize
import matplotlib


def plotSweepRes(Path,ax=None,idx=2,fit=False,xlabel="Frequency [Hz]",color='bo',xscale=1,yscale=1,ylabel=""):
	if ax==None:
		fig, ax = plt.subplots(1,1,figsize=(7,5))
	y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
	ax.xaxis.set_major_formatter(y_formatter)

	A = np.loadtxt("{root}\\Data.csv".format(root=Path),delimiter=',')
	ax.plot(A[:,0]*xscale,A[:,idx]*yscale,color)
	f = np.linspace(min(A[:,0]),max(A[:,0]),100)
	x0=A[np.argmax(A[:,idx]),0]
	A0=np.max(A[:,idx])
	if fit:
		p0=[x0,13,A0*.9,x0,20,A0*.1,0]
		p1,fp = scipy.optimize.curve_fit(lambda x,*p: m.Lorentz(x,*p[:3])+m.Gauss(x,*p[3:6])+p[6], A[:,0],A[:,idx],p0)
		ax.axhline(p1[6],linestyle='--')
		ax.plot(f,m.Lorentz(f,*p1[:3])+m.Gauss(f,*p1[3:6])+p1[6],'r');
		ax.plot(f,m.Lorentz(f,*p1[:3]),'g');
		ax.plot(f,m.Gauss(f,*p1[3:6])+p1[6],'m--');
		ax.axvline(p1[0],color='r',linestyle='--');
		ax.annotate("{:.0f}Hz".format(p1[0]),(p1[0],1.1*min(A[:,idx])));
	ax.set_xlabel(xlabel)
	if ylabel!="": ax.set_ylabel(ylabel)
