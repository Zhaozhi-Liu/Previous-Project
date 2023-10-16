import pandas as pd

def bisearch_right(a, x, lo=0, hi=None):
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = lo + (hi-lo)//2
        if x < a[mid]: hi = mid
        else: lo = mid+1
    return lo


def longest_dinterval(df):
    n = len(df)
    start = 0
    for i in range(n):
        if i<n-1 and df["d2"].values[i]<=df["d2"].values[i+1]:
            break
    return [start,i]

def longest_iinterval(df):
    n = len(df)
    ans = 0
    start = n-1
    for i in range(n-1,-1,-1):
        if i>0 and df["d2"].values[i]>=df["d2"].values[i-1]:
            break
    return [i,start]