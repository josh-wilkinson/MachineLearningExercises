import pandas as pd


def get_investigated_features(dataset):
    """
    Function to get the features listed in the description above.
    
    Input: dataset (pandas DataFrame) -- initial full listings dataset

    Output: dataset (pandas DataFrame) with only the features listed in the description above
    """

    features = [
        "name",
        "description",
        "neighborhood_overview",
        "property_type",
        "room_type",
        "amenities",
        "price",
        "accommodates",
        "bedrooms",
        "beds",
        "bathrooms"
    ]

    return dataset[features]


def feature_target_split(dataset):
    """
    Function to split the dataset into features and target variable.
    
    Input: dataset (pandas DataFrame) -- dataset with features and target variable

    Output: X (pandas DataFrame) -- features, y (pandas Series) -- target variable
    """
    
    X = dataset.drop(columns=["price"])
    y = dataset["price"]
    
    return X, y