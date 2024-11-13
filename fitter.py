import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import models 

V = []
T = []
V_error = []

def V_fit_func(V,T, p1, p2, p3):
    V = p1**p2 * T**p3
    return V 


V_curve_model = models.Model(V_fit_func)

V_fit_result = V_curve_model.fit(V=V, T=T, weights = 1 / V_error, p1 = 1, p2 = 0.5, p3 = -0.5)

V_fit_result
