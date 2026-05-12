import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("small_data/qq_Z.csv")

def invariant_mass(e1, e2):
    E  = e1["E"] + e2["E"]
    Px = e1["Px"] + e2["Px"]
    Py = e1["Py"] + e2["Py"]
    Pz = e1["Pz"] + e2["Pz"]
    m2 = E**2 - Px**2 - Py**2 - Pz**2
    return np.sqrt(np.maximum(m2, 0))

masses = []

for event_id, event in df.groupby("event_number"):

    e_minus = event[event["particle_id"] == 11]
    e_plus  = event[event["particle_id"] == -11]

    # skip bad events
    if len(e_minus) == 0 or len(e_plus) == 0:
        continue

    # vectorized cross product (NO loops)
    m_e = e_minus[["E","Px","Py","Pz"]].values
    p_e = e_plus[["E","Px","Py","Pz"]].values

    for me in m_e:
        E1, Px1, Py1, Pz1 = me

        for pe in p_e:
            E2, Px2, Py2, Pz2 = pe

            E  = E1 + E2
            Px = Px1 + Px2
            Py = Py1 + Py2
            Pz = Pz1 + Pz2

            m2 = E*E - Px*Px - Py*Py - Pz*Pz
            masses.append(np.sqrt(max(m2, 0)))

plt.hist(masses, bins=100, range=(0,200))
plt.xlabel("m_ee [GeV]")
plt.ylabel("Events")
plt.title("Dielectron invariant mass")
plt.grid()
plt.savefig("m_ee.png")
