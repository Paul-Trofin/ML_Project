# ML Classification of Particle Data (XGBoost)

This project performs a physics analysis of simulated proton-proton collision events, focusing on:

- reconstruction of the Z → e⁻e⁺ decay channel  
- comparison with background processes (tt̄, gg → tt̄)  
- machine learning classification using gradient boosted decision trees (XGBoost)

The goal is to study how well kinematic and event-level features can separate electroweak Z production from top-quark backgrounds.

---

## Physics Overview

We analyze simulated LHC-like events and reconstruct:

- invariant mass of electron pairs
- angular correlations (Δφ, Δη, ΔR)
- transverse momentum observables
- event activity (jets, MET, HT)

The Z boson signal is expected to appear as a peak around 91 GeV.
Verify this:
```bash
python3 plot_mass.py
```

---

Build the event-level dataset:
```bash
python3 build_dataset.py
```
This outputs the "events_dataset.csv" used to train the model. Here the kinematic variables are calculated, and e-e+ are stored as pairs.

---

Train the model:
```bash
python3 train_MLmodel.py
```
This trains an XGBoost classifier using the kinamtic variables.
It outputs:
- xgboost_model.json (trained model you can use)
- plots/ directory where you can visualize the ROC curve, the ML score and feature importance

---

## Now you can use the trained model as you like!
```bash
python3 apply_MLmodel.py
```
Just do not forget to change the "new_events.csv" with your actual database.

