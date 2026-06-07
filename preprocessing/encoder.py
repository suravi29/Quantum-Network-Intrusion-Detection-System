from sklearn.preprocessing import LabelEncoder


def encode_features(data):

    categorical_columns = [
        "protocol_type",
        "service",
        "flag"
    ]

    encoders = {}

    for column in categorical_columns:
        encoder = LabelEncoder()
        data[column] = encoder.fit_transform(data[column])

        encoders[column] = encoder

    return data, encoders