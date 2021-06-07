import numpy as np

get_data = lambda x : np.random.rand(x) #returns random data of shape `x`

e_tau = lambda a, b, c, d, e, f : a + b + c + d - e - f #follow order from Obj. function (above)