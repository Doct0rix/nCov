import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

from lxml import html
import requests
import datetime

days = 150
deathrate = .0175

def func(x, a, b, c):
     #return 1 / (1 + np.exp(-k*(x-x0)))
    
	 #return np.array(a) * np.exp(np.array(b) * x) + np.array(c)
	 return a * x**2 + b * x + c

page = requests.get('https://www.cdc.gov/coronavirus/2019-ncov/cases-updates/total-cases-onset.json')
tree = html.fromstring(page.content)
data = page.json()


total_cases_web = tree.xpath("//*[@id='cdc-chart-1-data']")

#total_cases_web[0] = total_cases_web[0].replace('Total cases: ', '')
#total_cases_web[0] = total_cases_web[0].replace(',', '')


#for range(1, len(data['data']['columns'][1])

#f = open("cases.data", "r")
#new = f.readline()
#newcases = [int(x) for x in new.split(",")]
#total = f.readline()
#totalcases = [int(x) for x in total.split(",")]
#date = f.readline()
#year, month, day = [int(x) for x in date.split(" ")]
#f.close()

#print(newcases)

totalcases = data['data']['columns'][1][38:]

for i in range(0, len(totalcases)):
  totalcases[i] = int(totalcases[i])

print(totalcases)

new = [1]
for i in range(1, len(totalcases)):
  new.append(totalcases[i] - totalcases[i-1])

newcases = new

print new

ydata = np.array(newcases)
ydataTotal = np.array(totalcases)

print(len(totalcases))
print(len(ydataTotal))

#lastdate = datetime.datetime(year, month, day).date()

#if total_cases > totalcases[-1]:
#  if datetime.datetime.now().date() > lastdate:
#    ydataTotal = np.append(ydataTotal, total_cases)
#    ydata = np.append(ydata, total_cases - totalcases[-1])
#  else:
#    totalcases[-1] = total_cases
#    newcases[-1] = totalcases[-1] - totalcases[-2]
#    ydata = np.array(newcases)
#    ydataTotal = np.array(totalcases)

#f = open("cases.data", "w")
#total_string = np.array2string(ydata, separator=' ', max_line_width=-1)
#for x in ydata:
#  f.write(str(x))
#  if x != ydata[len(ydata) - 1]:
#    f.write(",")

#f.write("\n")

#for x in ydataTotal:
#  f.write(str(x))
#  if x != ydataTotal[len(ydataTotal) - 1]:
#    f.write(",")

#f.write("\n")

#currentdate = datetime.datetime.now()
#f.write(currentdate.strftime("%Y %m %d"))

#f.close()

#if datetime.datetime.now().date() > lastdate:
#  if total_cases > totalcases[-1]:
#    ydataTotal = np.append(ydataTotal, total_cases)
#    ydata = np.append(ydata, total_cases - totalcases[-1])
#elif total_cases > totalcases[-1]:
#  totalcases[-1] = total_cases
#  newcases[-1] = total_cases[-1] - total_cases[-2]
#  ydata = np.array(newcases)
#  ydataTotal = np.array(totalcases)

xpredict = np.array(list(range(0,days)))
#ydata = np.array([8, 6, 23, 19, 31, 68, 40, 149, 117, 250, 270, 328, 388, 529, 599, 802, 959, 1511, 2884, 4978, 4862, 7347, 7285, 10075, 10008, 14693, 16437, 17620, 20395, 18207, 22291])
#ydataTotal = np.array([24, 30, 53, 72, 103, 171, 211, 360, 477, 727, 997, 1325, 1713, 2242, 2841, 3643, 4602, 6113, 8997, 13975, 18837, 26184, 33469, 43544, 53552, 68245, 84682, 102302, 122697, 140904, 163195])
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
