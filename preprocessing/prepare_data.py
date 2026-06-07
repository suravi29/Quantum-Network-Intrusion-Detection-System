from sklearn.preprocessing import LabelEncoder


def prepare_features_and_labels(data):

    # Remove original attack names
    data = data.drop("label", axis=1)

    # Encode attack categories
    category_encoder = LabelEncoder()

    data["attack_category"] = category_encoder.fit_transform(
        data["attack_category"]
    )

    # Separate features and target
    X = data.drop("attack_category", axis=1)
    y = data["attack_category"]

    return X, y, category_encoder