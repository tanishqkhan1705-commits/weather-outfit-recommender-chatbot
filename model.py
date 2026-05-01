
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─────────────────────────────
# STEP 1: DATASET
# ─────────────────────────────

rng = np.random.default_rng(seed=42)

WEATHER_OPTIONS = ["Clear", "Clouds", "Rain", "Snow"]
OCCASION_OPTIONS = ["casual", "office", "party"]

# ─────────────────────────────
# OUTFIT LOGIC
# ─────────────────────────────

def assign_outfit(temp, weather, occasion):

    # 🔥 HOT WEATHER
    if temp >= 30:

        if occasion == "casual":
            return "Oversized T-shirt, shorts & sneakers"

        elif occasion == "office":
            return "Formal cotton shirt, formal trousers & loafers"

        elif occasion == "party":
            return "Stylish summer party wear with accessories"

    # 🔥 WARM WEATHER
    elif temp >= 22:

        if occasion == "casual":
            return "T-shirt & jeans"

        elif occasion == "office":
            return "Blazer with formal pants"

        elif occasion == "party":
            return "Trendy club outfit"

    # 🔥 COOL WEATHER
    elif temp >= 15:

        if occasion == "casual":
            return "Shirt & jeans"

        elif occasion == "office":
            return "Formal shirt with blazer"

        elif occasion == "party":
            return "Stylish evening outfit"

    # 🔥 COLD WEATHER
    elif temp >= 8:

        if occasion == "casual":
            return "Sweater & jeans"

        elif occasion == "office":
            return "Formal winter jacket"

        elif occasion == "party":
            return "Winter party outfit"

    # 🔥 VERY COLD
    else:

        if occasion == "casual":
            return "Heavy coat & boots"

        elif occasion == "office":
            return "Formal heavy overcoat"

        elif occasion == "party":
            return "Luxury winter outfit"


# ─────────────────────────────
# ACCESSORIES LOGIC
# ─────────────────────────────

def assign_accessories(temp, weather):

    if weather in ["Rain", "Snow"]:
        return "Umbrella"

    elif temp >= 30:
        return "Sunglasses"

    elif temp <= 10:
        return "Gloves"

    else:
        return "None"


# ─────────────────────────────
# GENERATE DATA
# ─────────────────────────────

n = 1000

temps = rng.integers(2, 42, size=n).astype(float)
temps = temps + rng.uniform(-0.5, 0.5, size=n)
temps = temps.round(1)

weathers = rng.choice(
    WEATHER_OPTIONS,
    size=n,
    p=[0.4, 0.3, 0.2, 0.1]
)

occasions = rng.choice(
    OCCASION_OPTIONS,
    size=n
)

df = pd.DataFrame({
    "temp": temps,
    "weather": weathers,
    "occasion": occasions
})

df["outfit"] = [
    assign_outfit(t, w, o)
    for t, w, o in zip(df["temp"], df["weather"], df["occasion"])
]

df["accessories"] = [
    assign_accessories(t, w)
    for t, w in zip(df["temp"], df["weather"])
]

print(df.head())

# ─────────────────────────────
# STEP 2: FEATURES & TARGETS
# ─────────────────────────────

X = df[["temp", "weather", "occasion"]]

y_outfit = df["outfit"]
y_accessories = df["accessories"]

# ─────────────────────────────
# STEP 3: TRAIN TEST SPLIT
# ─────────────────────────────

X_train, X_test, y_train_o, y_test_o = train_test_split(
    X,
    y_outfit,
    test_size=0.2,
    random_state=42
)

X_train2, X_test2, y_train_a, y_test_a = train_test_split(
    X,
    y_accessories,
    test_size=0.2,
    random_state=42
)

# ─────────────────────────────
# STEP 4: PREPROCESSING
# ─────────────────────────────

cat_cols = ["weather", "occasion"]

preprocessor = ColumnTransformer([
    (
        "cat",
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ),
        cat_cols
    )
], remainder="passthrough")

# ─────────────────────────────
# STEP 5: RANDOM FOREST MODELS
# ─────────────────────────────

outfit_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=5,
        random_state=42
    ))
])

accessories_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=5,
        random_state=42
    ))
])

# ─────────────────────────────
# STEP 6: TRAIN
# ─────────────────────────────

outfit_model.fit(X_train, y_train_o)
accessories_model.fit(X_train2, y_train_a)

# ─────────────────────────────
# STEP 7: TEST
# ─────────────────────────────

pred_o = outfit_model.predict(X_test)
pred_a = accessories_model.predict(X_test2)

print("\nOUTFIT ACCURACY:")
print(accuracy_score(y_test_o, pred_o))

print("\nACCESSORIES ACCURACY:")
print(accuracy_score(y_test_a, pred_a))

# ─────────────────────────────
# STEP 8: SAMPLE PREDICTIONS
# ─────────────────────────────

samples = pd.DataFrame([
    {
        "temp": 35,
        "weather": "Clear",
        "occasion": "party"
    },

    {
        "temp": 35,
        "weather": "Clear",
        "occasion": "office"
    },

    {
        "temp": 35,
        "weather": "Clear",
        "occasion": "casual"
    }
])

print("\nSAMPLE PREDICTIONS:\n")

for i, row in samples.iterrows():

    sample_df = pd.DataFrame([row])

    outfit = outfit_model.predict(sample_df)[0]
    accessory = accessories_model.predict(sample_df)[0]

    print(f"INPUT: {dict(row)}")
    print(f"OUTFIT: {outfit}")
    print(f"ACCESSORIES: {accessory}")
    print("-" * 50)

# ─────────────────────────────
# STEP 9: SAVE MODELS
# ─────────────────────────────

with open("outfit_model.pkl", "wb") as f:
    pickle.dump(outfit_model, f)

with open("accessories_model.pkl", "wb") as f:
    pickle.dump(accessories_model, f)

print("\n✅ Models saved successfully!")
