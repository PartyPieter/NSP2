import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import models

begin_t = 0.0041
T_above = [20 ,20.1, 47.7, 39.8,40,39,38.5,38,32.2,31.6,29.3]
T_under = [20 ,19.9,45.0, 36.3, 35.2, 34.7, 34.1, 33.7, 31.9, 30.9, 29.1]
Temperatures = []
Temp_err = []

print(len(T_above))
df = pd.read_csv("temps.csv", sep = ",")
T_above = df.iloc[:,0]
T_under = df.iloc[:,1]

print(len(T_above))
for i in range(len(T_above)):
   
  Temperatures.append(0.5 * (T_above[i] + T_under[i])) 
  Temp_err.append(0.5 * 0.002 * (T_above[i] + T_under[i]) + 1)

x = 0.06
x_err = 0.002
#v = [1500]
#v_err = [x_err * 2 / (0.12 / 1500) ]
v = []
v_err = []
df = []

cut_value = 8

# executing speed calculations for every measurement
aantal_metingen = 62
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
    t  = (time[cut_values[0]] - begin_t) / 1000 #+- (8 * 10 **(-5) + (2 + 2/3) * 10 ** (-6))

    print(t)

    v.append((2 * x) / t)
    v_err.append(2 * x_err / t)

    # plot time-amplitude curve
 #   plt.plot(time, Amplitude)
 #   plt.plot(time[cut_values[0]], Amplitude[cut_values[0]], 'ro', markersize = 5)
 #   plt.xlabel("t (ms) ")
 #   plt.ylabel("Amp")
#    plt.title("time - amplitude curve")
 #   plt.show()

print(len(Temperatures))
print(len(v))
# plotting V-T plot
plt.plot(Temperatures, v, 'bo')
plt.xlabel("Temperature (C) ")
plt.ylabel("Speed (m/s)")
plt.title("Speed of sound through water at given temperature ")
plt.show()

# TO BE PERFECTED
def V_fit_func(T, p1, p2, p3, p4):
  #  V = p1**p2 * T**p3
    v = p1 + p2 * T + p3 * T**2 + p4 * T**3
    return v 


# fitting in motion
V_curve_model = models.Model(V_fit_func)
weights = np.array([1 / err for err in v_err])
V_fit_result = V_curve_model.fit(v, T=Temperatures, weights = weights, xerr = Temp_err, p1 = 1, p2 = -0.5, p3 = -0.5, p4 = 1)

# print and plot fit
V_fit_result.plot(numpoints = 1000)
plt.errorbar(Temperatures, v, xerr = Temp_err, yerr = None, fmt = 'none' )
plt.show()
print(V_fit_result.fit_report())
