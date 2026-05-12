import pandas as pd
from xgboost import XGBClassifier

##################################################
# LOAD TRAINED MODEL
##################################################

model = XGBClassifier()
model.load_model("xgboost_model.json")

##################################################
# LOAD NEW EVENTS
##################################################

df = pd.read_csv("new_events.csv")

features = [
    "invariant_mass",
    "delta_phi_ee",
    "delta_eta_ee",
    "delta_r_ee",
    "leading_electron_pt",
    "subleading_electron_pt",
    "HT",
    "n_jets",
    "MET"
]

X = df[features]

##################################################
# PREDICT
##################################################

scores = model.predict_proba(X)[:,1]

df["ML_score"] = scores

print(df.head())

df.to_csv("scored_events.csv", index=False)
