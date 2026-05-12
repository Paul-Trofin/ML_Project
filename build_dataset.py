import pandas as pd
import numpy as np

############################################################
###################### LOAD DATA ###########################
############################################################

df_signal = pd.read_csv("small_data/qq_Z.csv")
df_bkg1   = pd.read_csv("small_data/qq_tt.csv")
df_bkg2   = pd.read_csv("small_data/gg_tt.csv")

############################################################
###################### LABELS ##############################
############################################################

df_signal["label"] = 1
df_bkg1["label"]   = 0
df_bkg2["label"]   = 0

############################################################
###################### MERGE ###############################
############################################################

df = pd.concat(

    [df_signal, df_bkg1, df_bkg2],
    ignore_index=True
)

############################################################
###################### KINEMATICS ##########################
############################################################

df["pt"] = np.sqrt(
    df["Px"]**2 + df["Py"]**2
)

df["eta"] = np.arcsinh(
    df["Pz"] / (df["pt"] + 1e-12)
)

df["phi"] = np.arctan2(
    df["Py"],
    df["Px"]
)

############################################################
###################### HELPERS #############################
############################################################

def delta_phi(e1, e2):

    dphi = np.abs(
        e1["phi"] - e2["phi"]
    )

    return np.minimum(
        dphi,
        2*np.pi - dphi
    )

def delta_eta(e1, e2):

    return np.abs(
        e1["eta"] - e2["eta"]
    )

def delta_r(e1, e2):

    deta = delta_eta(e1, e2)

    dphi = delta_phi(e1, e2)

    return np.sqrt(
        deta**2 + dphi**2
    )

def invariant_mass(e1, e2):

    E  = e1["E"]  + e2["E"]
    Px = e1["Px"] + e2["Px"]
    Py = e1["Py"] + e2["Py"]
    Pz = e1["Pz"] + e2["Pz"]

    m2 = E**2 - Px**2 - Py**2 - Pz**2

    return np.sqrt(max(m2, 0))

############################################################
###################### EVENT LOOP ##########################
############################################################

event_features = []

grouped = df.groupby("event_number")

for event_id, event in grouped:

    ########################################################
    # ELECTRONS ONLY
    ########################################################

    electrons = event[
        (event["particle_id"] == 11) |
        (event["particle_id"] == -11)
    ]

    e_minus = electrons[
        electrons["particle_id"] == 11
    ]

    e_plus = electrons[
        electrons["particle_id"] == -11
    ]

    ########################################################
    # REQUIRE e- AND e+
    ########################################################

    if len(e_minus) == 0 or len(e_plus) == 0:
        continue

    ########################################################
    # BEST Z CANDIDATE
    ########################################################

    best_mass       = None
    best_dphi       = None
    best_deta       = None
    best_dr         = None
    best_zpt        = None

    leading_pt      = None
    subleading_pt   = None

    best_diff = 1e9

    ########################################################
    # LOOP OVER e- e+ PAIRS
    ########################################################

    for _, e1 in e_minus.iterrows():

        for _, e2 in e_plus.iterrows():

            ################################################
            # INVARIANT MASS
            ################################################

            m = invariant_mass(e1, e2)

            if m < 1:
                continue

            ################################################
            # ANGULAR VARIABLES
            ################################################

            dphi = delta_phi(e1, e2)

            deta = delta_eta(e1, e2)

            dr = delta_r(e1, e2)

            ################################################
            # Z BOSON pT
            ################################################

            z_px = e1["Px"] + e2["Px"]
            z_py = e1["Py"] + e2["Py"]

            z_pt = np.sqrt(
                z_px**2 + z_py**2
            )

            ################################################
            # BEST Z CANDIDATE
            ################################################

            diff = abs(m - 91.2)

            if diff < best_diff:

                best_diff = diff

                best_mass = m
                best_dphi = dphi
                best_deta = deta
                best_dr   = dr
                best_zpt  = z_pt

                pts = sorted(

                    [e1["pt"], e2["pt"]],

                    reverse=True
                )

                leading_pt = pts[0]

                subleading_pt = pts[1]

    ########################################################
    # SAFETY CHECK
    ########################################################

    if best_mass is None:
        continue

    ########################################################
    # EVENT VARIABLES
    ########################################################

    # scalar HT
    HT = event["pt"].sum()

    # jet multiplicity
    n_jets = (
        event["particle_name"] == "jet"
    ).sum()

    ########################################################
    # MISSING TRANSVERSE ENERGY
    ########################################################

    met_x = event["Px"].sum()
    met_y = event["Py"].sum()

    MET = np.sqrt(
        met_x**2 + met_y**2
    )

    ########################################################
    # LABEL
    ########################################################

    label = event["label"].iloc[0]

    ########################################################
    # STORE EVENT
    ########################################################

    event_features.append({

        ####################################################
        # Z candidate variables
        ####################################################

        "invariant_mass": best_mass,

        "delta_phi_ee": best_dphi,

        "delta_eta_ee": best_deta,

        "delta_r_ee": best_dr,

        "z_pt": best_zpt,

        ####################################################
        # electron variables
        ####################################################

        "leading_electron_pt":
            leading_pt,

        "subleading_electron_pt":
            subleading_pt,

        ####################################################
        # event variables
        ####################################################

        "HT": HT,

        "n_jets": n_jets,

        "MET": MET,

        ####################################################
        # target
        ####################################################

        "label": label
    })

############################################################
###################### OUTPUT ##############################
############################################################

events_df = pd.DataFrame(event_features)

print(events_df.head())

print("\nColumns:")
print(events_df.columns)

print("\nLabel distribution:")
print(events_df["label"].value_counts())

############################################################
###################### SAVE CSV ############################
############################################################

events_df.to_csv(

    "events_dataset.csv",

    index=False
)

print("\nSaved dataset: events_dataset.csv")
