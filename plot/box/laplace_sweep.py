import os
import sys
import numpy
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_data import *
from plot.box.paths_laplace import *

def laplace_eval(path, eval_key):
    res_dict = loading_info_all(path, eval_key=eval_key)
    print(dict(sorted(res_dict.items(), key=lambda item: item[1],reverse=True)))

laplace_eval(pdrand_laplace_test0, "Dynamic awareness = ")
laplace_eval(pdrand_laplace_test1, "Averaged improvement = ")