import numpy as np
import scipy

def Lorentz(x,x0,G,Amp=1):
	return Amp*(.5*G)**2/((x-x0)**2+(.5*G)**2)

def Gauss(x,x0,sig,Amp=1,norm=False):
	if norm:
		Amp=1/(sig*np.sqrt(2*np.pi))
	return Amp*np.exp(-(x-x0)**2/(2*sig**2))

def ShiftDetect(A,B):
	Corr = np.fft.fftshift(np.real(np.fft.ifft2(np.conj(np.fft.fft2(A))*np.fft.fft2(B))))
	c = np.unravel_index(Corr.argmax(), Corr.shape)
	return Corr,(c[1]-Corr.shape[1]/2,c[0]-Corr.shape[0]/2)

def normalize(x):
	return (x-np.min(x))/(np.max(x)-np.min(x))

def CDF(x,mu,sig):
	return .5*(1+scipy.special.erf((x-mu)/(sig*np.sqrt(2))))

def deconv(source,target):
	count_offset = (np.max(target)+np.min(target))/2.
	source = 2*source-1
	target = target - count_offset
	target = pySPM.tukeyfy(target, 0.08)
	return np.fft.fftshift(np.real(np.fft.fft2(target)/np.fft.fft2(source)))

def deconv_tik(source,target,mu):
	count_offset = (np.max(target)+np.min(target))/2.
	source = 2*source-1
	target = target - count_offset
	tf = np.fft.fft2(source)
	tf /= np.size(tf)
	recon_tf = np.conj(tf) / ( np.abs(tf)**2 + mu) 
	work = np.fft.fftshift( np.real( np.fft.ifft2( np.fft.fft2(target) * recon_tf )))
	return work

def getProfile_deconv_tik(source,target,mu,ax=None):
	work=abs(deconv_tik(source,target,mu))
	mid = int(work.shape[0]//2)
	P=work[mid,:]
	M=np.max(P)
	a,b  = getFWHM(P)
	if ax is not None:
		ax.plot(np.linspace(-30,29,60),P[mid-30:mid+30],'ro',label="PSF profile")
		ax.axhline(M)
		ax.axhline(M/2)
		ax.axvline(a-mid)
		ax.axvline(b-mid)
		ax.set_title("mu={:.2e}, FWHM={:f}".format(mu,b-a))
	return work,b-a
	
def smiley(size,r=.9):
	g = np.zeros((size,size))
	x = np.linspace(-1,1,size)
	X,Y = np.meshgrid(x,x)
	R = np.sqrt(X**2+Y**2)

	g[R<r] =1
	g[((X+.3*r/.9)**2+(Y+.4*r/.9)**2)<.02*r/.9] = 0
	g[((X-.3*r/.9)**2+(Y+.4*r/.9)**2)<.02*r/.9] = 0
	g[(R>.4444*r)*(R<.66666*r)*(Y>0)] = 0
	return g

def square(size, size1):
	S = np.zeros((size,size))
	S[int((size-size1)//2):int((size+size1)//2),int((size-size1)//2):int((size+size1)//2)]=1
	return S

def getFWHM(P):
	M=np.max(P)
	a2=np.argmax(P>M/2)
	a1=a2-1
	a=a1+(M/2-P[a1])/(P[a2]-P[a1])
	b2=np.argmax((P<M/2)*(np.arange(P.size)>a1))
	b1=b2-1
	b=b2-(M/2-P[b2])/(P[b1]-P[b2])
	return a,b
	
def zoom_center(A,dx,dy=None):
	assert type(dx) in [int,float]
	if dy is None:
		dy=dx
	H, W = A.shape
	x1,x2 = int(W/2-dx),int(W/2+dx)
	y1,y2 = int(H/2-dy),int(H/2+dy)
	return A[y1:y2,x1:x2]

def fact(x):
	assert type(x) is int
	if x < 0:
		x=-x
	if x<2: return [x]
	f=[]
	for i in range(2,int(np.sqrt(x)+1)):
		while x%i==0:
			f.append(i)
			x/=i
	if x>1:
		f.append(x)
	return f
