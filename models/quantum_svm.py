from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def train_quantum_svm(
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\nCreating Quantum Feature Map...")

    feature_map = ZZFeatureMap(
        feature_dimension=X_train.shape[1],
        reps=2
    )

    quantum_kernel = FidelityQuantumKernel(
        feature_map=feature_map
    )

    print("Computing Quantum Kernel Matrix...")

    kernel_train = quantum_kernel.evaluate(
        x_vec=X_train
    )

    kernel_test = quantum_kernel.evaluate(
        x_vec=X_test,
        y_vec=X_train
    )

    model = SVC(
        kernel="precomputed"
    )

    print("Training Quantum SVM...")

    model.fit(
        kernel_train,
        y_train
    )

    predictions = model.predict(
        kernel_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\nQuantum SVM Accuracy:")
    print(round(accuracy * 100, 2), "%")

    return model, predictions