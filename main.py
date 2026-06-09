from preprocessing.data_loader import load_dataset
from preprocessing.attack_mapping import map_attack
from preprocessing.encoder import encode_features
from preprocessing.prepare_data import prepare_features_and_labels
from preprocessing.scaler import scale_features
from preprocessing.split_data import split_dataset
from preprocessing.feature_reduction import reduce_features

from models.classical_svm import train_classical_svm
from models.quantum_svm import train_quantum_svm

from sklearn.model_selection import train_test_split


# =====================================
# Load Dataset
# =====================================

train_data = load_dataset(
    "dataset/KDDTrain+.txt"
)

# Remove difficulty column
train_data.drop(
    "difficulty",
    axis=1,
    inplace=True
)

# =====================================
# Attack Mapping
# =====================================

train_data["attack_category"] = (
    train_data["label"]
    .apply(map_attack)
)

print("\nAttack Category Distribution:")
print(
    train_data["attack_category"]
    .value_counts()
)

# =====================================
# Encode Features
# =====================================

train_data, encoders = encode_features(
    train_data
)

# =====================================
# Prepare Features and Labels
# =====================================

X, y, category_encoder = (
    prepare_features_and_labels(
        train_data
    )
)

print("\nFeature Matrix Shape:")
print(X.shape)

print("\nTarget Vector Shape:")
print(y.shape)

print("\nCategory Mapping:")
print(
    dict(
        zip(
            category_encoder.classes_,
            category_encoder.transform(
                category_encoder.classes_
            )
        )
    )
)

# =====================================
# Feature Scaling
# =====================================

X_scaled, scaler = scale_features(
    X
)

print("\nScaled Feature Matrix Shape:")
print(X_scaled.shape)

# =====================================
# Train/Test Split
# =====================================

(
    X_train,
    X_test,
    y_train,
    y_test
) = split_dataset(
    X_scaled,
    y
)

print("\nTraining Set Shape:")
print(X_train.shape)

print("\nTesting Set Shape:")
print(X_test.shape)

# =====================================
# PCA Feature Reduction
# =====================================

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

print(
    "\nExplained Variance:"
)

print(
    round(
        pca.explained_variance_ratio_.sum()
        * 100,
        2
    ),
    "%"
)

# =====================================
# Classical SVM Dataset
# =====================================

(
    X_train_small,
    _,
    y_train_small,
    _
) = train_test_split(
    X_train_reduced,
    y_train,
    train_size=10000,
    random_state=42,
    stratify=y_train
)

(
    X_test_small,
    _,
    y_test_small,
    _
) = train_test_split(
    X_test_reduced,
    y_test,
    train_size=2000,
    random_state=42,
    stratify=y_test
)

# =====================================
# Classical SVM
# =====================================

svm_model, predictions = (
    train_classical_svm(
        X_train_small,
        y_train_small,
        X_test_small,
        y_test_small
    )
)

# =====================================
# Quantum SVM Dataset
# =====================================

(
    X_train_quantum,
    _,
    y_train_quantum,
    _
) = train_test_split(
    X_train_reduced,
    y_train,
    train_size=300,
    random_state=42,
    stratify=y_train
)

(
    X_test_quantum,
    _,
    y_test_quantum,
    _
) = train_test_split(
    X_test_reduced,
    y_test,
    train_size=100,
    random_state=42,
    stratify=y_test
)

# =====================================
# Quantum SVM
# =====================================

quantum_model, quantum_predictions = (
    train_quantum_svm(
        X_train_quantum,
        y_train_quantum,
        X_test_quantum,
        y_test_quantum
    )
)