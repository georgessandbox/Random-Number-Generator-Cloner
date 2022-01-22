import random
import pandas as pd
import numpy as np
from scipy.stats import truncnorm
from scipy.stats import nakagami
import matplotlib.pylab as plt
from sklearn import datasets, linear_model


df = pd.read_csv("lotto.csv")


def get(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs()


frequencytable = {}

for (idx, row) in df.iterrows():
    for x in row:
        
        if x in frequencytable:
            frequencytable[x] += 1
        else:
            frequencytable[x] = 1
print(frequencytable)

# sorted by key, return a list of tuples
lists = sorted(frequencytable.items())

x, y = zip(*lists)  # unpack a list of pairs into two tuples

plt.scatter(x, y)


regr = linear_model.LinearRegression()
regr.fit(np.array(list(frequencytable.keys())).reshape(-1, 1), list(frequencytable.values()))

y = regr.predict(np.arange(1, 50).reshape(-1, 1))

plt.plot(np.arange(1, 50), y, 'g^')
plt.xlim(1, 50)
plt.grid()
plt.show()



""" 
while True:

    
    val1, val2, val3, val4,\
    val5, val6, val7 = random.randint(1, 33), random.randint(2, 37), random.randint(3, 42), random.randint(5, 45), \
    random.randint(8, 48), random.randint(17, 49), random.randint(18, 50)
    

    # Random based on normal on mean and std of sample of each column
    val1 = round(np.random.normal(6, 5.126157803))
    val2 = round(np.random.normal(13, 6.784745003))
    val3 = round(np.random.normal(19, 7.546174206))
    val4 = round(np.random.normal(25, 7.741738221))
    val5 = round(np.random.normal(31, 7.488497465 ))
    val6 = round(np.random.normal(37, 6.828599061 ))
    val7 = round(np.random.normal(44, 5.374320074 ))
  
    # Random based on normal and min and max
    val1 = round(get(mean=3, sd=5, low=1, upp=10))
    val2 = round(get(mean=8, sd=6, low=2, upp=26))
    val3 = round(get(mean=18, sd=7, low=3, upp=33))
    val4 = round(get(mean=25, sd=7, low=5, upp=41))
    val5 = round(get(mean=32, sd=7, low=16, upp=45))
    val6 = round(get(mean=37, sd=25, low=22, upp=49))
    val7 = round(get(mean=43, sd=13, low=36, upp=50))

    a = [val1, val2, val3, val4,val5, val6, val7]
    b = [4, 6, 7, 10, 17, 27, 44]

   # if (df[['#1','#2', '#3', '#4', '#5', '#6', '#7']] == a).all(1).any():
    if a == b:
        print(a)
        print("wow")
        break
#see how frequent is pairs 8,9 """
