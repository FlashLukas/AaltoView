"""TR-MOKE with a 100 MHz laser: undo the conjugation of the negative-alias frequencies (complex lock-in data)."""

from aaltoview.loading import tr_moke_unfold

NAME = "TR-MOKE unfold (100 MHz laser)"


def load(ds, path):
    return tr_moke_unfold(ds, f_rep_mhz=100.0)
