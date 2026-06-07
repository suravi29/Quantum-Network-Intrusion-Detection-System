from sklearn.preprocessing import MinMaxScaler


def scale_features(X):

    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler