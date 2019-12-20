import pandas as pd
import random


data = pd.read_csv("./p_test.csv", error_bad_lines=False)

count_pos = 0
count_neg = 0

nrows = len(data.iloc[:, 0])
drop_rows = []
for i in range(nrows):
    if i % 10000 == 0:
        print(str(i/nrows*100)[:5] + "%")
    if data.loc[i, 'Labels'] == 0:
        if random.randint(1,2) == 1:
            drop_rows.append(i)


nrows = len(data.iloc[:, 0])

for i in range(nrows):
    if data.loc[i, 'Labels'] == 0:
        count_neg += 1
    else:
        count_pos += 1
data = data.drop(drop_rows)

print("Neg Samples:", str((count_neg - len(drop_rows)) / (nrows - len(drop_rows)) * 100))
print("Pos Samples:", str(count_pos / (nrows - len(drop_rows)) * 100))


# data.to_csv("bid_training.csv")