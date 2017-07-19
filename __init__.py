from pyScholi.plot import *
from pyScholi.math import *
from pyScholi.science import *
from pyScholi.colors import *
from pyScholi.aa import *
from pyScholi import AFM
import os
__all__ = ["aa","AFM","science","colors","math","plot","os"]

cloud = os.path.join(os.getenv("HOMEPATH"),"ownCloud")
cloudT = os.path.join(cloud, "ToFSIMS")
cloudDB = os.path.join(os.getenv("HOMEPATH"),"Dropbox")