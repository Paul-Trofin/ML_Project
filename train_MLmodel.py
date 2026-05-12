import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from xgboost import XGBClassifier
import matplotlib.pyplot as plt

############################################################
###################### LOAD DATA ###########################
############################################################

events_df = pd.read_csv("events_dataset.csv")

os.makedirs("plots", exist_ok=True)

############################################################
###################### FEATURES ############################
############################################################

features = [

    # Z candidate physics
    "invariant_mass",
    "delta_phi_ee",
    "delta_eta_ee",
    "delta_r_ee",

    # electron kinematics
    "leading_electron_pt",
    "subleading_electron_pt",

    # event activity
    "HT",
    "n_jets",
    "MET"
]

X = events_df[features]
y = events_df["label"]

############################################################
###################### STRATIFIED SPLIT ####################
############################################################

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,
    random_state=42,
    stratify=y
)

############################################################
###################### CLASS BALANCE #######################
############################################################

signal_count = (y_train == 1).sum()
background_count = (y_train == 0).sum()

if signal_count == 0:
    raise ValueError("No signal events in training set.")

scale_pos_weight = background_count / signal_count

print("\nSignal:", signal_count)
print("Background:", background_count)
print("Scale Pos Weight =", scale_pos_weight)

############################################################
###################### MODEL ###############################
############################################################

model = XGBClassifier(

    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,

    subsample=0.8,
    colsample_bytree=0.8,

    min_child_weight=3,

    objective="binary:logistic",
    eval_metric="logloss",

    random_state=42,

    scale_pos_weight=scale_pos_weight
)

############################################################
###################### TRAIN ###############################
############################################################

model.fit(X_train, y_train)

############################################################
###################### PREDICTIONS #########################
############################################################

y_pred = model.predict(X_test)

############################################################
###################### EVALUATION ##########################
############################################################

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

############################################################
###################### ROC CURVE ###########################
############################################################

y_probs = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

print("\nROC AUC =", roc_auc)

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()

plt.savefig("plots/roc_curve.png", dpi=300)
plt.close()

############################################################
###################### FEATURE IMPORTANCE ##################
############################################################

importance = model.feature_importances_

feature_importance_df = pd.DataFrame({

    "feature": features,
    "importance": importance
})

feature_importance_df = feature_importance_df.sort_values(

    by="importance",
    ascending=False
)

print("\nFeature Importances:\n")
print(feature_importance_df)

############################################################
###################### PLOT IMPORTANCE #####################
############################################################

plt.figure(figsize=(8,6))

plt.barh(

    feature_importance_df["feature"],
    feature_importance_df["importance"]
)

plt.xlabel("Importance")
plt.title("Feature Importance")

plt.gca().invert_yaxis()

plt.grid()

plt.savefig("plots/feature_importance.png", dpi=300)
plt.close()

############################################################
###################### SCORE DISTRIBUTION ##################
############################################################

signal_scores = y_probs[y_test == 1]
background_scores = y_probs[y_test == 0]

plt.figure(figsize=(8,6))

plt.hist(
    background_scores,
    bins=50,
    alpha=0.6,
    density=True,
    label="Background"
)

plt.hist(
    signal_scores,
    bins=50,
    alpha=0.6,
    density=True,
    label="Signal"
)

plt.xlabel("ML Score")
plt.ylabel("Density")
plt.title("XGBoost Output Score")

plt.legend()
plt.grid()

plt.savefig("plots/ml_score.png", dpi=300)
plt.close()

############################################################
###################### SAVE MODEL ##########################
############################################################

model.save_model("xgboost_model.json")

print("\nModel saved: xgboost_model.json")
print("Plots saved in: plots/")
