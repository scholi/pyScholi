def printTable(t):
	Ncols=len(t[0])
	Lcol=[0]*Ncols
	for x in t:
		for i in range(Ncols):
			Lcol[i]=max(Lcol[i],len(repr(x[i])))
	for j,x in enumerate(t):
		print("  ".join([u"{:"+['.','_'][j%2]+"<"+str(Lcol[i]+4)+"}" for i in range(Ncols)]).format(*x))
	
