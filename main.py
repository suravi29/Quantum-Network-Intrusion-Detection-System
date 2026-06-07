from preprocessing.data_loader import load_dataset

# Load dataset
train_data = load_dataset("dataset/KDDTrain+.txt")

# Remove difficulty column
train_data.drop("difficulty", axis=1, inplace=True)

print(train_data.head())
print("\nShape:", train_data.shape)
print("\nUnique attack labels:")
print(train_data["label"].unique())