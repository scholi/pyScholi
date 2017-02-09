def hot2val(r,g,b):
    A=0.365079
    B=0.7460321
    return A*(r-0.0416)/0.9584+(B-A)*g+(1-B)*b

def hotImg2val(img):
	r=img[:,:,0]
	g=img[:,:,1]
	b=img[:,:,2]
	return hot2val(r,g,b)
