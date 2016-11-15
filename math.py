import numpy as np

def Lorentz(x,x0,G,Amp=1):
	return Amp*(.5*G)**2/((x-x0)**2+(.5*G)**2)

def Gauss(x,x0,sig,Amp=1):
	return Amp*np.exp(-(x-x0)**2/(2*sig**2))

def ShiftDetect(A,B):
	Corr = np.fft.fftshift(np.real(np.fft.ifft2(np.conj(np.fft.fft2(A))*np.fft.fft2(B))))
	c = np.unravel_index(Corr.argmax(), Corr.shape)
	return Corr,(c[1]-Corr.shape[1]/2,c[0]-Corr.shape[0]/2)

def normalize(x):
	return (x-np.min(x))/(np.max(x)-np.min(x))
