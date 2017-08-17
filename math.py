import numpy as np
import scipy

def Lorentz(x,x0,G,Amp=1):
    return Amp*(.5*G)**2/((x-x0)**2+(.5*G)**2)

def Gauss(x,x0,sig,Amp=1,norm=False):
    if norm:
        Amp=1/(sig*np.sqrt(2*np.pi))
    return Amp*np.exp(-(x-x0)**2/(2*sig**2))

def LG(x, x0, sig=None, Amp=1, lg=.5, FWHM=None):
    assert sig is not None or FWHM is not None
    if FWHM is None:
        FWHM = 2*np.sqrt(2*np.log(2))*sig
    if sig is None:
        sig = FWHM/(2*np.sqrt(2*np.log(2)))
    return Amp*((1-lg)*Gauss(x,x0,sig)+lg*Lorentz(x,x0,FWHM))

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
    work = abs(deconv_tik(source,target,mu))
    mid = int(work.shape[0]//2)
    P = work[mid,:]
    M = np.max(P)
    a,b  = getFWHM(P)
    if ax is not None:
        ax.plot(np.linspace(-30,29,60), P[mid-30:mid+30], 'ro', label="PSF profile")
        ax.axhline(M)
        ax.axhline(M/2)
        ax.axvline(a-mid)
        ax.axvline(b-mid)
        ax.set_title("mu={:.2e}, FWHM={:f}".format(mu, b-a))
    return work, b-a
    
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

def logistic(x, lower=0, upper=1, growth=1, x0=0, nu=1, C=1):
    return lower+(upper-lower)/(C+np.exp(-growth*(x-x0)))**(1/nu)
    
def circular_profile(Img, x0, y0, R, Rn=0, N=20, Astart=0, Astop=360, cmap='jet', prog=False, axImg=None, ax=None, width=1, upper=None, lower=None, axPolar=None):
    from matplotlib import colors, cm, gridspec
    import matplotlib.pyplot as plt
    import pySPM
    import scipy.optimize as opt
    if prog:
        from tqdm import tqdm_notebook as tqdm
    CM =  plt.get_cmap(cmap) 
    cNorm  = colors.Normalize(vmin=0, vmax=N)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=CM)

    # Sweep angle around x0,y0 and calulate sigma from a CDF fit of each profile
    sig = []
    angles = []
    def f(x, x0, s, A=None, bg=0):
        if upper is not None:
            A = upper-bg
        return bg + A*CDF(x, x0, s)
    def f2(x, x0, s, bg=0):
        A = upper-bg
        return bg + A*CDF(x, x0, s)
         
    
    if x0>Img.shape[1]:
        Astop = min(Astop, np.degrees(2*np.pi+np.arctan2(y0-Img.shape[0], Img.shape[1]-x0)-1e-3))
        Astart = max(Astart, np.degrees(np.arctan2(y0, Img.shape[1]-x0)-1e-3))
    #print("Limit angles", Astart, "->", Astop)
    if Astart%360 == Astop%360:
        ListAngles = np.arange(Astart, Astop, (Astop-Astart)/N)
    else:
        ListAngles = np.linspace(Astart, Astop, N)
    if prog:
        ListAngles = tqdm(ListAngles, leave=False)
   
    
    for i, angle in enumerate(ListAngles):
        a = np.radians(angle)
        angles.append(a)
        Ra = R
        Ren = Rn
        if x0 > Img.shape[1]:
            Ren = np.max([(Img.shape[1]-x0)/np.cos(a), Ren])
        elif x0<0:
            Ren = np.max([-x0/np.cos(a), Ren])
        if y0 > Img.shape[0] and np.sin(a)<0:
            Ren = np.max([(Img.shape[0]-y0)/np.sin(a), Ren])
        elif y0 < 0:
            Ren = np.max([-y0/np.sin(a), Ren])
        if x0 > 0 and x0 < Img.shape[1] and np.cos(a)>0:
            Ra = np.min([(Img.shape[1]-x0)/np.cos(a), Ra])
        elif x0 > 0 and x0 < Img.shape[1] and np.cos(a)<0:
            Ra = np.min([-x0 / np.cos(a),Ra])
        if y0 > 0 and np.sin(a)>0:
            Ra = np.min([y0 / np.sin(a),Ra])
        elif y0 < Img.shape[0] and np.sin(a)<0:
            Ra = np.min([(y0-Img.shape[0])/np.sin(a), Ra])
        #print(angle, a, Ra, Ren)
        l, p = pySPM.getProfile(Img, x0+Ren*np.cos(a), y0-Ren*np.sin(a),\
                                x0+Ra*np.cos(a), y0-Ra*np.sin(a),\
                                ax=axImg, width=width, color=scalarMap.to_rgba(i))
        profile = np.mean(p, axis=1)
        if upper is None:
            p0 = [l[len(l)//2], 10, np.max(p)-np.min(p),np.min(p) ]
            popt, pcov = opt.curve_fit(f, l, profile, p0)
        else:
            p0 = [l[len(l)//2], 10, np.min(p) ]
            popt, pcov = opt.curve_fit(f2, l, profile, p0)
   
        sig.append(popt[1])
        if ax:
            ax.plot((l-popt[0]), profile, color=scalarMap.to_rgba(i), linestyle=':')
            if upper is None:
                ax.plot((l-popt[0]), f(l,*popt), color=scalarMap.to_rgba(i))
            else:
                ax.plot((l-popt[0]), f2(l,*popt), color=scalarMap.to_rgba(i))
            ax.set_title("Profiles from point ({},{})".format(x0,y0))
    if Astop%360==Astart%360:
        angles.append(angles[0])
        sig.append(sig[0])
    if axPolar is not None:
        axPolar.plot(angles, sig, label="({}x{})".format(x0,y0))
    return angles, sig