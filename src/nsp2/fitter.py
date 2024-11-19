import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import models

begin_t = 0.0041
Temperatures = [41, 36]
x = 0.06
v = []
v_err = []
df = []

cut_value = 3

# executing speed calculations for every measurement
aantal_metingen = 2
for i in range(aantal_metingen):
   
    cut_values = []
    df = pd.read_csv("meting_" + str(i) + ".csv", sep = "\t")
    time = df.iloc[:,0]
    Amplitude = df.iloc[:,1]
    avg_amp = sum(Amplitude)/len(Amplitude)

    for i in range (len(Amplitude) - 1):
        
        # save points where amplitude is higher then average + cutvalue
        if Amplitude[i] > avg_amp + cut_value and time[i] > begin_t:
         cut_values.append(i)

    # time it takes for wave to move is time of first point where amplitude is high enough - time start of measurement
    t  = time[cut_values[0]] - begin_t

    print(t)
    print(cut_values[0])
    v.append(2 * x / t)
    #TO BE ISSUED
    v_err.append(3)

    # plot time-amplitude curve
    plt.plot(time, Amplitude)
    plt.plot(time[cut_values[0]], Amplitude[cut_values[0]], 'ro', markersize = 10)
    plt.xlabel("t (ms) ")
    plt.ylabel("Amp")
    plt.title("time - amplitude curve")
    plt.show()

print(v)

# plotting V-T plot
plt.plot(Temperatures, v, 'bo')
plt.xlabel("Temperature (C) ")
plt.ylabel("Speed (m/s)")
plt.title("Speed of sound through water at given temperature ")
plt.show()

# TO BE PERFECTED
def V_fit_func(T, p1, p3):
  #  V = p1**p2 * T**p3
    v = p1  +  p3 *( T)  
    return v 

print("yoyoyo")

# fitting in motion
V_curve_model = models.Model(V_fit_func)

weights = np.array([1 / err for err in v_err])
V_fit_result = V_curve_model.fit(v, T=Temperatures, weights = weights , p1 = 1, p3 = -0.5)

# print and plot fit
V_fit_result.plot(numpoints = 1000)
plt.ylim(1.6 , 1.7)
plt.show()
print(V_fit_result.fit_report())
