from preprocessing.data_loader import load_dataset

# Load training dataset
train_data = load_dataset("dataset/KDDTrain+.txt")

print(train_data.head())
print("\nShape:", train_data.shape)