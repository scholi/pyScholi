import numpy as np

def getBAM(y,y0,N=10,edges=False):
	P= y*0
	l=0
	Edges=[]
	for x in [(68,80)]*N+[(67,691),
			(691,293),(294,293),
			(380,19.5),
			(400,195),(195,195),
			(420,135),(135,135),
			(380,96),(96,96),
			(320,68),(68,68),
			(260,48),(49,48),
			(220,38),(39,38),
			(105,24),(24,24),
			(767,38),
			(494,3.6),
			(492,14.2)]:
		l+=x[0]
		Edges.append(y0+l)
		P = P + ((y-y0)>=l)*((y-y0)<=l+x[1])
		if y0+l>=y[0] and y0+l<=y[-1]:
			P[np.argmin(abs(y-y0-l))]=1 # At least 1 pixel set
		l+=x[1]
		Edges.append(y0+l)
	if edges: return P,Edges
	return P
