from sklearn.decomposition import PCA


def reduce_features(
    X_train,
    X_test,
    n_components=8
):

    pca = PCA(
        n_components=n_components,
        random_state=42
    )

    X_train_reduced = pca.fit_transform(X_train)

    X_test_reduced = pca.transform(X_test)

    return (
        X_train_reduced,
        X_test_reduced,
        pca
    )