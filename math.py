def Lorentz(x,x0,G,Amp=1):
	return Amp*(.5*G)**2/((x-x0)**2+(.5*G)**2)

def Gauss(x,x0,sig,Amp=1):
	return Amp*np.exp(-(x-x0)**2/(2*sig**2))
