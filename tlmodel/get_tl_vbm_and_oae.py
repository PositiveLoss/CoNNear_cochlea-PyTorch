import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from .cochlear_model import cochlea_model
from .solve_one_cochlea import solve_one_cochlea


def tl_vbm_and_oae(stim, L):
    """Run the transmission-line cochlea model for a stimulus batch."""
    sheraPo = np.loadtxt(Path("StartingPoles.dat"))
    irregularities = 1  # adaptIRR0!
    opts = {
        "sheraPo": sheraPo,
        "storeflag": "ve",
        "probe_points": "abr",
        "Fs": 100e3,
        "channels": np.min(stim.shape),
        "subjectNo": 1,
        "sectionsNo": int(1e3),
        "output_folder": os.getcwd() + "/",
        "numH": 13.0,
        "numM": 3.0,
        "numL": 3.0,
        "IrrPct": 0.05,
        "nl": "vel",
        "L": L,
    }

    irr_on = irregularities * np.ones((1, opts["channels"])).astype("int")
    cochlear_list = [
        [cochlea_model(), stim[i], irr_on[0][i], i, opts]
        for i in range(opts["channels"])
    ]

    print("running human auditory model 2018: Verhulst, Altoe, Vasilkov")
    with mp.Pool(mp.cpu_count(), maxtasksperchild=1) as pool:
        output = pool.map(solve_one_cochlea, cochlear_list)

    print("cochlear simulation: done")

    return output
