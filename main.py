import pandas as pd

train_data = pd.read_csv(
    "dataset/KDDTrain+.txt",
    header=None
)

print(train_data.head())
print(train_data.shape)