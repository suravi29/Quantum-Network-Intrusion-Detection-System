from preprocessing.data_loader import load_dataset
from preprocessing.attack_mapping import map_attack
from preprocessing.encoder import encode_features
from preprocessing.prepare_data import prepare_features_and_labels
from preprocessing.scaler import scale_features
from preprocessing.split_data import split_dataset
from preprocessing.feature_reduction import reduce_features

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

# Split dataset
X_train, X_test, y_train, y_test = split_dataset(
    X_scaled,
    y
)

# Reduce features using PCA
(
    X_train_reduced,
    X_test_reduced,
    pca
) = reduce_features(
    X_train,
    X_test,
    n_components=8
)

print("\nReduced Training Shape:")
print(X_train_reduced.shape)

print("\nReduced Testing Shape:")
print(X_test_reduced.shape)

print("\nExplained Variance:")
print(
    round(
        pca.explained_variance_ratio_.sum() * 100,
        2
    ),
    "%"
)

print("\nTraining set shape:")
print(X_train.shape)

print("\nTesting set shape:")
print(X_test.shape)

print("\nTraining labels:")
print(y_train.shape)

print("\nTesting labels:")
print(y_test.shape)

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