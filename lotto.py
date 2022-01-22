import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.stats import truncnorm

# todo : try to shuffle instead of randomize
# todo : implement gradient boost / neuralnetwork solutions

df = pd.read_csv("lotto.csv")

# Helper function from stackoverflow to get a random number with specific standard deviation, min, and max


def get(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs()


# Generate numbers that fall within the analysis to train the model on
new_rows = []
for i in range(1, 12000):
    val1, val2, val3, val4,\
        val5, val6, val7 = random.randint(1, 33), random.randint(2, 37), random.randint(3, 42), random.randint(5, 45), \
        random.randint(8, 48), random.randint(17, 49), random.randint(18, 50)

    row = {'#1': val1,  '#2': val2,  '#3': val3,  '#4': val4,
           '#5': val5,  '#6': val6,  '#7': val7,  'class': 0}

    a = [val1, val2, val3, val4, val5, val6, val7, 1]

    if not (df == a).all(1).any():
        new_rows.append(row)
    else:
        print("wow")

df = df.append(new_rows, ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)  # shuffle
print("matrix size : " + str(df.shape) )

""" 
X_train, X_test, y_train, y_test = train_test_split(
    df[['#1', '#2', '#3', '#4', '#5', '#6', '#7']], df['class'], test_size=1)
 """

# Fitting the model
logistic = LogisticRegression()
logistic.fit(df[['#1', '#2', '#3', '#4', '#5', '#6', '#7']], df['class'])

# keep guessing until one shows up
while True:
    """ val1 = round(get(mean=6, sd=5.126157803, low=1, upp=33))
    val2 = round(get(mean=13, sd=6.784745003, low=2, upp=37))
    val3 = round(get(mean=19, sd=7.546174206, low=3, upp=42))
    val4 = round(get(mean=25, sd=7.741738221, low=5, upp=45))
    val5 = round(get(mean=31, sd=7.488497465, low=8, upp=48))
    val6 = round(get(mean=37, sd=6.828599061, low=17, upp=49))
    val7 = round(get(mean=44, sd=5.374320074, low=18, upp=50)) """
    val1 = round(get(mean=3, sd=5, low=1, upp=7))
    val2 = round(get(mean=8, sd=6, low=2, upp=26))
    val3 = round(get(mean=18, sd=7, low=3, upp=33))
    val4 = round(get(mean=25, sd=7, low=5, upp=41))
    val5 = round(get(mean=32, sd=7, low=16, upp=45))
    val6 = round(get(mean=37, sd=25, low=22, upp=49))
    val7 = round(get(mean=43, sd=13, low=36, upp=50))

    # test with auto generated numbers or numbers of your own
    oneitem = [[val1, val2, val3, val4,  val5, val6, val7]]
    #oneitem = [[4, 6, 7, 10, 17, 27, 44]]

    if len(set(oneitem[0])) == len(oneitem[0]):

        y = logistic.predict(oneitem)

        if y == 1:
            print(str(oneitem) + "This one!")
            break
        else:
            print(str(oneitem) + " nope")
            continue
    else:
            print(str(oneitem) + " nope")

# one random value to guess on
    """ 
    val1, val2, val3, val4,\
        val5, val6, val7 = random.randint(1, 33), random.randint(2, 37), random.randint(3, 42), random.randint(5, 45), \
        random.randint(8, 48), random.randint(17, 49), random.randint(18, 50) """

# make val unique
    """ if not (val1 == val2 == val3 == val4 == val5 == val6 == val7): #wrong way lol
        oneitem = pd.DataFrame({'#1': [val1],  '#2': [val2],  '#3': [val3],  '#4': [val4],
                                '#5': [val5],  '#6': [val6],  '#7': [val7]})
                                 """

# guess on a bag on items
""" 
test_bag = []
for i in range(1, 100000000):
    val1, val2, val3, val4,\
        val5, val6, val7 = random.randint(1, 33), random.randint(2, 37), random.randint(3, 42), random.randint(5, 45), \
                           random.randint(8, 48), random.randint(17, 49), random.randint(18, 50)
    row = {'#1': val1,  '#2': val2,  '#3': val3,  '#4': val4,
           '#5': val5,  '#6': val6,  '#7': val7}
    test_bag.append(row)

onemore = {'#1': 2,  '#2': 8,  '#3': 16,  '#4': 17,
        '#5': 19,  '#6':  26,  '#7': 42}
test_bag.append(onemore)

test_bag_df = pd.DataFrame(data=test_bag)

test_bag_df["classPredicted"] = logistic.predict(test_bag_df)

print(test_bag_df[test_bag_df['classPredicted'] == 1] )
 """
