import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import models

begin_t = 0.0041
x = 0.06
v = []
df = []
#df = pd.read_csv("real_test_meting.csv", sep = "\t")


#V_error = []

#df["V_err"] = df["V"] * 0.001 + 0.044
cut_value = 2
cut_values = []

aantal_metingen = 2

for i in range(aantal_metingen):

    df = pd.read_csv("meting_" + str(i) + ".csv", sep = "\t")
    time = df.iloc[:,0]
    Amplitude = df.iloc[:,1]

    for i in range (len(Amplitude) - 1):
        delta_amp = np.abs(Amplitude[i + 1] - Amplitude[i])
        print(delta_amp)
        if delta_amp > cut_value and time[i] > 0.02:
         cut_values.append(i)

    t  = time[cut_values[0]] - begin_t
    print(t)
    print(cut_values[0])
    v.append(2 * x / t)
    plt.plot(time, Amplitude)
    plt.plot(time[cut_values[0]], Amplitude[cut_values[0]], 'ro', markersize = 10)
    plt.show()

print(v)


"""
def V_fit_func(T, p1, p2, p3):
  #  V = p1**p2 * T**p3
    V = p1  + (p2 * T) ** p3
    return V 

print("yoyoyo")

V_curve_model = models.Model(V_fit_func)

V_fit_result = V_curve_model.fit(df["V"], T=df["T"], weights = 1 / df["V_err"], p1 = 1, p2 = 2, p3 = -0.5)


V_fit_result.plot(numpoints = 1000)
plt.show()
V_fit_result
"""