from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def train_classical_svm(
    X_train,
    y_train,
    X_test,
    y_test
):

    svm_model = SVC(
        kernel="rbf",
        random_state=42
    )

    print("Training Classical SVM...")

    svm_model.fit(
        X_train,
        y_train
    )

    predictions = svm_model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\nAccuracy:")
    print(round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    return svm_model, predictions