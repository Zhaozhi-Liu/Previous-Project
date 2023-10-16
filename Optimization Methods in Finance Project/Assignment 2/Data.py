import pandas as pd
import numpy as np
from scipy.stats.mstats import gmean

df = pd.DataFrame(pd.read_csv('indices2.csv'))
df_copy = pd.DataFrame(pd.read_csv('indices2.csv'))
Index = list(df.columns)
del Index[0]

# Question 1
# Calculate return rate
for j in Index:
    for i in range(364):
        df.loc[i+1,j]=df_copy.loc[i+1,j]/df_copy.loc[i,j]
df =df.drop(labels=0)

# group by quarter
Q1 = df.loc[df['Date'].str.contains('Jan|Feb|Mar')]
Q2 = df.loc[df['Date'].str.contains('Apr|May|Jun')]
Q3 = df.loc[df['Date'].str.contains('Jul|Aug|Sep')]
Q4 = df.loc[df['Date'].str.contains('Oct|Nov|Dec')]

# Batch quarter
def get_return(datas):
    rr = np.array([gmean(datas.loc[df['Date'].str.contains('2013')].iloc[:,1:])])
    for year in range(2014,2020):
        geometricMean = np.array([gmean(datas.loc[df['Date'].str.contains(f'{year}')].iloc[:,1:])])
        rr = np.row_stack((rr,geometricMean))
    return pd.DataFrame(data = rr,index = list(range(2013,2020)),columns=Index)

Q1_returnrates = get_return(Q1)
Q2_returnrates = get_return(Q2)
Q3_returnrates = get_return(Q3)
Q4_returnrates = get_return(Q4)

Q1_returnrates.to_csv("Q1_returnrates.csv", index_label="year")
Q2_returnrates.to_csv("Q2_returnrates.csv", index_label="year")
Q3_returnrates.to_csv("Q3_returnrates.csv", index_label="year")
Q4_returnrates.to_csv("Q4_returnrates.csv", index_label="year")

Q1.to_csv("Q1_raw.csv", index_label="year")
Q2.to_csv("Q2_raw.csv", index_label="year")
Q3.to_csv("Q3_raw.csv", index_label="year")
Q4.to_csv("Q4_raw.csv", index_label="year")

df.to_csv("df.csv", index_label="year")
