from preprocessing.data_loader import load_dataset
from preprocessing.attack_mapping import map_attack
from preprocessing.encoder import encode_features
from preprocessing.prepare_data import prepare_features_and_labels
from preprocessing.scaler import scale_features

# Load dataset
train_data = load_dataset("dataset/KDDTrain+.txt")

# Remove difficulty column
train_data.drop("difficulty", axis=1, inplace=True)

# Convert attack labels into categories
train_data["attack_category"] = train_data["label"].apply(map_attack)

# Display class distribution
print("\nAttack category distribution:")
print(train_data["attack_category"].value_counts())

# Encode categorical features
train_data, encoders = encode_features(train_data)

# Create feature matrix and target vector
X, y, category_encoder = prepare_features_and_labels(train_data)

# Scale features
X_scaled, scaler = scale_features(X)

print("\nScaled feature matrix shape:")
print(X_scaled.shape)

print("\nTarget vector shape:")
print(y.shape)

print("\nFeature matrix shape:")
print(X.shape)

print("\nTarget vector shape:")
print(y.shape)

print("\nAttack category encoding:")
print(dict(zip(
    category_encoder.classes_,
    category_encoder.transform(category_encoder.classes_)
)))

# Display first 5 rows after encoding
print("\nFirst 5 rows after encoding:")
print(train_data.head())

# Display dataset shape
print("\nDataset Shape:")
print(train_data.shape)