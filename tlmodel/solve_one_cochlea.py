import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def solve_one_cochlea(model):
    """Solve one configured cochlea model."""
    coch = model[0]
    opts = model[4]

    sheraPo = opts["sheraPo"]
    probe_points = opts["probe_points"]
    Fs = opts["Fs"]
    subjectNo = opts["subjectNo"]
    sectionsNo = opts["sectionsNo"]
    IrrPct = opts["IrrPct"]
    nl = opts["nl"]

    coch.init_model(
        model[1],
        Fs,
        sectionsNo,
        probe_points,
        Zweig_irregularities=model[2],
        sheraPo=sheraPo,
        subject=subjectNo,
        IrrPct=IrrPct,
        non_linearity_type=nl,
    )
    coch.solve()
    return {
        "fs_bm": Fs,
        "v": coch.Vsolution,
        "e": coch.oto_emission,
        "cf": coch.cf,
    }
