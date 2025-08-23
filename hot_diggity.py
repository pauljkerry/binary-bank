import pandas as pd

combine = pd.DataFrame()
CATS = []
NUMS = []
for c in combine.columns[:-1]:
    t = "CAT"
    if combine[c].dtype == 'object':
        CATS.append(c)
    else:
        NUMS.append(c)
        t = "NUM"
    n = combine[c].nunique()
    na = combine[c].isna().sum()
    print(f"[{t}] {c} has {n} unique and {na} NA")
print("CATS:", CATS)
print("NUMS:", NUMS)

QuantileDMatrix