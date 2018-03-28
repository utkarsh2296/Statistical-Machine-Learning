import numpy as np
import pandas as pd

whole = pd.read_csv('C:\\Users\Vaibhav\PycharmProjects\SML\SMLProject\data.csv', encoding='latin', low_memory=False)
null_columns = whole.columns[whole.isnull().any()]
count = 0
for i in null_columns:
    if whole[i].isnull().sum() > 20000:
        whole = whole.drop(i, axis=1)
        print("dropped")
        count += 1

print(count)
print(whole.shape)
print(whole.columns)

# mat = whole.as_matrix()
# print(type(mat))

