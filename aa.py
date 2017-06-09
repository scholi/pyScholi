def printTable(t):
    Ncols = len(t[0])
    Lcol = [0]*Ncols
    for x in t:
        for i in range(Ncols):
            Lcol[i] = max(Lcol[i],len(repr(x[i])))
    for j,x in enumerate(t):
        print("  ".join([u"{:"+['.', '_'][j%2]+"<"+str(Lcol[i]+4)+"}" for i in range(Ncols)]).format(*x))

def htmlTable(t, show=True, header=False):
    import html
    s = "<table><tr>"
    if header:
        s += "<th>"+"</th><th>".join([html.escape(str(y)) for y in t[0]])+"</th></tr><tr>"
        t = t[1:]
    s += "".join([\
        "<tr><td>"+"</td><td>".join([\
            html.escape(str(y)) for y in x]) + "</td></tr>" for x in t])+"</table>"
    if show:
        from IPython.display import display, HTML
        display(HTML(s))
    else:
        return s