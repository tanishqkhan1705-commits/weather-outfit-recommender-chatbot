import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, precision_score

# ─────────────────────────────
# STEP 1: DATA BANAO
# ─────────────────────────────
np.random.seed(42)

WEATHER_OPTIONS  = ["Clear", "Clouds", "Rain", "Snow"]
OCCASION_OPTIONS = ["casual", "office", "party"]

def assign_outfit(temp, weather, occasion):
    if temp >= 30:
        base = {"casual": "T-shirt & shorts", "office": "Light shirt & trousers", "party": "Stylish light outfit"}
    elif temp >= 22:
        base = {"casual": "T-shirt & jeans", "office": "Formal light wear", "party": "Trendy outfit"}
    elif temp >= 15:
        base = {"casual": "Shirt & jeans", "office": "Formal shirt & pants", "party": "Stylish shirt"}
    elif temp >= 8:
        base = {"casual": "Sweater & jeans", "office": "Formal jacket", "party": "Casual trendy outfit"}
    else:
        base = {"casual": "Heavy coat & boots", "office": "Coat & trousers", "party": "Heavy jacket"}

    if weather == "Rain":
        if occasion == "casual":   return "Light jacket & jeans"
        elif occasion == "office": return "Formal jacket"
        else:                      return base[occasion]
    if weather == "Snow":
        return "Heavy coat & boots" if occasion == "casual" else "Coat & trousers"
    return base[occasion]

def assign_accessories(temp, weather):
    if weather in ["Rain", "Snow"]:       return "Umbrella"
    if temp >= 25 and weather == "Clear": return "Sunglasses"
    return "None"

# ─────────────────────────────
# LARGER DATASET — 2000 samples
# Har class mein zyada samples → model generalize karega
# ─────────────────────────────
n = 2000

temps     = np.random.randint(2, 42, size=n).astype(float) + np.random.uniform(-0.5, 0.5, size=n)
temps     = temps.round(1)
weathers  = np.random.choice(WEATHER_OPTIONS,  size=n, p=[0.4, 0.25, 0.25, 0.1])
occasions = np.random.choice(OCCASION_OPTIONS, size=n)

outfits      = [assign_outfit(t, w, o)   for t, w, o in zip(temps, weathers, occasions)]
accessories  = [assign_accessories(t, w) for t, w   in zip(temps, weathers)]

# ─────────────────────────────
# LABEL NOISE ADD KARO
# Real world mein data perfect nahi hota
# 8% rows mein random outfit swap karenge
# ─────────────────────────────
OUTFIT_CLASSES = list(set(outfits))
noise_idx = np.random.choice(n, size=int(0.08 * n), replace=False)
for i in noise_idx:
    outfits[i] = np.random.choice(OUTFIT_CLASSES)

df = pd.DataFrame({
    "temp":        temps,
    "weather":     weathers,
    "occasion":    occasions,
    "outfit":      outfits,
    "accessories": accessories,
})
print(f"Dataset: {df.shape[0]} rows | {df['outfit'].nunique()} outfit classes")

# ─────────────────────────────
# STEP 2: TRAIN TEST SPLIT
# ─────────────────────────────
X = df[["temp", "weather", "occasion"]]
y_outfit      = df["outfit"]
y_accessories = df["accessories"]

X_train, X_test, y_train_o, y_test_o, y_train_a, y_test_a = train_test_split(
    X, y_outfit, y_accessories,
    test_size=0.2,
    random_state=42,
    stratify=y_outfit
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────
# STEP 3: PIPELINE
# ─────────────────────────────
cat_cols = ["weather", "occasion"]

def make_pipeline(clf):
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ), cat_cols)
    ], remainder="passthrough")
    return Pipeline([("preprocessor", pre), ("classifier", clf)])

# ─────────────────────────────
# STEP 4: MODELS
# max_depth thoda relax kiya — 8 → data complex hua
# min_samples_leaf add kiya → leaf pe bhi min samples enforce
# ─────────────────────────────
model_outfit = make_pipeline(
    RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=42
    )
)

model_accessories = make_pipeline(
    RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=42
    )
)

# ─────────────────────────────
# STEP 5: TRAIN
# ─────────────────────────────
model_outfit.fit(X_train, y_train_o)
model_accessories.fit(X_train, y_train_a)

# ─────────────────────────────
# STEP 6: METRICS
# ─────────────────────────────
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model_outfit, X_train, y_train_o, cv=cv, scoring="accuracy")
pred_o    = model_outfit.predict(X_test)

print("\n" + "="*50)
print("OUTFIT MODEL — ALL METRICS")
print("="*50)
print(f"CV Accuracy  : {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
print(f"Test Accuracy: {accuracy_score(y_test_o, pred_o):.3f}")
print(f"Precision    : {precision_score(y_test_o, pred_o, average='weighted', zero_division=0):.3f}")
print(f"Recall       : {recall_score(y_test_o, pred_o, average='weighted', zero_division=0):.3f}")
print(f"F1 Score     : {f1_score(y_test_o, pred_o, average='weighted', zero_division=0):.3f}")
print("\nPer-class Report:")
print(classification_report(y_test_o, pred_o, zero_division=0))

pred_a = model_accessories.predict(X_test)
print("="*50)
print("ACCESSORIES MODEL — ALL METRICS")
print("="*50)
print(f"Accuracy : {accuracy_score(y_test_a, pred_a):.3f}")
print(f"Precision: {precision_score(y_test_a, pred_a, average='weighted'):.3f}")
print(f"Recall   : {recall_score(y_test_a, pred_a, average='weighted'):.3f}")
print(f"F1 Score : {f1_score(y_test_a, pred_a, average='weighted'):.3f}")
print(classification_report(y_test_a, pred_a))

# ─────────────────────────────
# STEP 7: SAMPLE PREDICTIONS
# ─────────────────────────────
print("="*50)
print("SAMPLE PREDICTIONS")
print("="*50)
samples = pd.DataFrame([
    {"temp": 30, "weather": "Clear",  "occasion": "party"},
    {"temp": 10, "weather": "Rain",   "occasion": "office"},
    {"temp": 20, "weather": "Clouds", "occasion": "casual"},
    {"temp":  3, "weather": "Snow",   "occasion": "casual"},
])
for i, row in samples.iterrows():
    o = model_outfit.predict(pd.DataFrame([row]))[0]
    a = model_accessories.predict(pd.DataFrame([row]))[0]
    print(f"  {row['temp']}C | {row['weather']} | {row['occasion']}")
    print(f"  → Outfit: {o} | Accessories: {a}\n")

# ─────────────────────────────
# STEP 8: PICKLE SAVE
# ─────────────────────────────
with open("outfit_model.pkl",      "wb") as f: pickle.dump(model_outfit,      f)
with open("accessories_model.pkl", "wb") as f: pickle.dump(model_accessories, f)
print("outfit_model.pkl + accessories_model.pkl saved!")




