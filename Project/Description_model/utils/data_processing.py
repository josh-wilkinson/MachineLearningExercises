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
        #"property_type",
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

def process_amenities(dataset):
    """
    Function to process the amenities feature in the dataset.
    
    Input: dataset (pandas DataFrame) -- dataset with amenities feature
    Output: dataset (pandas DataFrame) with processed amenities feature

    
    """

    split = dataset["amenities"].str.strip('[]') # remove the brackets
    split = split.str.replace(' ', '')  # remove the space between the amenities
    split = split.str.replace('"', '')  # remove the space between the amenities
    split = split.str.replace('\\u2013', '')
    split = split.str.replace('\\u2019', '')
    split = split.str.split(pat = ',')  # split the string into a list of amenities
    split = split.apply(lambda amenities: [amenity.lower() for amenity in amenities])

    substitutions = {
    "tv": "TV",
    "soundsystem":"SoundSystem",
    'kitchen': 'Kitchen',
    'soap': 'Soap',
    'shampoo': 'Shampoo',
    'refrigerator': 'Refrigerator',
    'stove': 'Stove',
    'oven': 'Oven',
    'wifi': 'WiFi',
    'ethernetconnection':"WiFi",
    'washer': 'Washer',
    'gym': 'Gym',
    'airconditioning': 'AC',
    'ac-': 'AC',
    'conditioner': 'HairConditioner',
    'housekeeping': 'Housekeeping',
    'wardrobe': 'Wardrobe',
    'grill': 'Grill',
    'pool': 'Pool',
    'backyard': 'Backyard',
    'parking': 'Parking',
    'garage': 'Parking',
    'fridge': 'Refrigerator',
    'dryer': 'Dryer',
    'gameconsole': 'GameConsole',
    'nintendowii': 'GameConsole',
    'ps3': 'GameConsole',
    'standalonehighchair': 'HighChair',
    'baby': 'BabyAmenities',
    'children': 'BabyAmenities',
    'highchair':'BabyAmenities',
    'changingtable': 'BabyAmenities',
    'crib': 'BabyAmenities',
    'exerciseequipment': 'ExerciseEquipment',
    'treadmill': 'ExerciseEquipment',
    'yogamat': 'ExerciseEquipment',
    'workoutbench': 'ExerciseEquipment',
    'freeweights': 'ExerciseEquipment',
    'coffeemaker': 'CoffeeMaker',
    'coffeemachine': 'CoffeeMaker',
    'espressomachine': 'CoffeeMaker',
    'indoorfireplace': 'IndoorFireplace',
    'hottub': 'HotTub',
    'balcony': 'Balcony',
    'clothingstorage': 'ClothingStorage',
    'closet': 'ClothingStorage',
    'safe': 'Safe',
    'netflix':'VideoStreaming',
    'amazonprimevideo': 'VideoStreaming',
    'hbomax': 'VideoStreaming',
    'disney+': 'VideoStreaming',
    'hulu': 'VideoStreaming',
    'view':'GoodView', #park, beack city, etc.
    'charger': 'Charger',
    'elevator': 'Elevator',
    'sauna': 'Sauna',
    'smoke': 'SafetyEquipment',
    'lock': 'SafetyEquipment',
    'alarm': 'SafetyEquipment',
    'securitycamera': 'SafetyEquipment',
    'noisedecibelmonitorsonproperty':"SafetyEquipment",
    'fireextinguisher': 'SafetyEquipment',
    "fireplaceguards": 'SafetyEquipment',
    'showergel': 'ShowerGel',
    'douchegel': 'ShowerGel',
    'douche': 'ShowerGel',
    'heat': 'Heating',
    'beachaccess': 'BeachAccess',

    
    }

    remove = [
        "yearold",  # e.g. "1 year old"
        "aweek",      # e.g. "newly renovated"
        "differentbrands",
        "denon",
        "dove",
        "everyday",
        "fa",
        "friday",
        "nivea",
        "hpneutral",
        "saturday",
        "smeg",
        "sonos",
        "tyllsn",
        "wednesday",
        "thursday",
        "jbl",
    ]
    for old, new in substitutions.items():
        split = split.apply(lambda amenities: [new if old in amenity else amenity for amenity in amenities])

    for remove_tag in remove:
        split = split.apply(lambda amenities: [amenity for amenity in amenities if remove_tag not in amenity])

    split = split.apply(lambda amenities: [amenity if amenity in substitutions.values() else "Other" for amenity in amenities])
    tags = set(split.explode().values)

    # substitute the column in the train dataframe
    dataset["amenities"] = split


    # create a new Boolean column for each tag
    for tag in tags:
        dataset[tag] = [tag in dataset['amenities'].loc[i] for i in dataset.index]

    #drop the amenities column
    dataset.drop(columns=["amenities"], inplace=True)

    return dataset


