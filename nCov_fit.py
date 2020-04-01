import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
days = 150
deathrate = .0175

def func(x, a, b, c):
     #return 1 / (1 + np.exp(-k*(x-x0)))
    
	 #return np.array(a) * np.exp(np.array(b) * x) + np.array(c)
	 return a * x**2 + b * x + c
xpredict = np.array(list(range(0,days)))
ydata = np.array([8, 6, 23, 19, 31, 68, 40, 149, 117, 250, 270, 328, 388, 529, 599, 802, 959, 1511, 2884, 4978, 4862, 7347, 7285, 10075, 10008, 14693, 16437, 17620, 20395, 18207, 22291])
ydataTotal = np.array([24, 30, 53, 72, 103, 171, 211, 360, 477, 727, 997, 1325, 1713, 2242, 2841, 3643, 4602, 6113, 8997, 13975, 18837, 26184, 33469, 43544, 53552, 68245, 84682, 102302, 122697, 140904, 163195])
ydatalog = np.log(ydata)
xdataTotal = np.array(list(range(0,len(ydataTotal))))
xdata = np.array(list(range(0,len(ydata))))
params, pcov = curve_fit(func, xdata, ydatalog)
print(params)
predictfit = func(xpredict, *params)
predictfithigh = func(xpredict, *(1.1*params))
predictfitlow = func(xpredict, *(0.9*params))
fit = func(xdata, *params) 
fitlog = np.log(fit)

#plt.plot(xpredict, predictfithigh, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(params))
#plt.plot(xpredict, predictfitlow, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(params))
#plt.plot(xdata, fit, 'r-', label='fit: x0=%5.3f, k=%5.3f' % tuple(params))
print(predictfit) 
ydatapredict = np.exp(predictfit)
# ydatapredicthigh = np.exp(predictfithigh)
# ydatapredictlow = np.exp(predictfitlow)
print(ydatapredict)
# logtotalCases = np.zeros(days)
totalCases = np.zeros(days)
# totalCaseshigh = np.zeros(days)
# totalCaseslow = np.zeros(days)
for i in range(0, days):
	totalCases[i] = totalCases[i-1] + ydatapredict[i]
	# totalCaseshigh[i] = totalCases[i-1] + ydatapredicthigh[i]
	# totalCaseslow[i] = totalCases[i-1] + ydatapredictlow[i]

print(totalCases)
max = np.amax(predictfit)
print(max)
print(np.argmax(predictfit))
print(np.argmax(predictfit) - len(xdata))
print(np.amax(totalCases))
print(deathrate * np.amax(totalCases))
#plotting
plt.figure()
#total
plt.subplot(211)
plt.plot(xpredict, totalCases)
plt.plot(xdataTotal, ydataTotal)
plt.xlabel('')
plt.ylabel('Total Infected')
#new
plt.subplot(212)
plt.plot(xpredict, predictfit, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(params))
plt.plot(xdata, ydatalog, 'b-', label='data')
plt.xlabel('day')
plt.ylabel('ln(New Cases)')
plt.legend()
plt.show()

# plt.plot(xpredict, totalCaseshigh)
# plt.plot(xpredict, totalCaseslow)
