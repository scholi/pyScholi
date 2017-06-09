from IPython.display import display, HTML, Image
import base64
import os

def show_peoples(Persons):
    L = []
    s = "<table><tr><th>First Name</th><th>Name</th><th>Email</th><th>Abt.</th><th>Photo</th></tr>"
    f = open("Q://DBTelefon/TelWin.dat","r")
    for x in f:
        line = x.strip().split(';')
        for p in Persons:
            if line[0] == p.split(",")[0].strip() and line[1] == p.split(",")[1].strip():
                L.append(line)
                Persons.remove(p)
                try:
                    fi = open("Q://Inex/Portrait/{3}{5}-portrait-ipdef.jpg".format(*line),"rb")
                    img = base64.b64encode(fi.read())
                    fi.close()
                except:
                    try:
                        fi = open("Q://Inex/Portrait_ohne_Abteilungsnummer/{3}-portrait-ipdef.jpg".format(*line),"rb")
                        img = base64.b64encode(fi.read())
                        fi.close()
                    except:
                        found = False
                        for i in os.listdir('Q://Inex/Portrait/'):
                            if i.startswith(line[3]):
                                fi = open("Q://Inex/Portrait/"+i,"rb")
                                img = base64.b64encode(fi.read())
                                fi.close()
                                found = True
                                break
                        if not found:
                            img = b""
                img = "<img alt=\"Image not found ("+line[3]+")\" src=\"data:image/png;base64,"+img.decode('ascii')+"\" class=\"photo\"/>"
                s+=("<tr class=\"abt{5}\"><td>{1}</td><td>{0}</td><td>{18}</td>"\
                "<td>{5}</td><td>"+img+"</td></tr>").\
                format(*line)
    for x in Persons:
        s += "<tr class=\"abtUnkn\"><td>{1}</td><td>{0}</td><td>Unknown</td><td>Unknown</td><td></td>Unknown</tr>".format(*x.split(", "))
    s += "</table>"
    f.close()
    display(HTML(s))
    return L