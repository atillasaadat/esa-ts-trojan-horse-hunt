# --- visualise_trigger.py  ---------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def visualise_trigger(trig_np: np.ndarray, model_id: int, out_dir: Path, show: bool = False):
    """
    trig_np  : (HORIZON, C) numpy array returned by optimise_trigger()
    model_id : int identifier (e.g. 17)
    out_dir  : where to save PNG
    show     : True → plt.show(); False → just save
    """

    out_dir.mkdir(parents=True, exist_ok=True)   # <‑‑ NEW LINE

    H, C = trig_np.shape
    fig, axes = plt.subplots(2, 1, figsize=(10,6), gridspec_kw={"height_ratios":[2,1]})
    ch_names = ["44","45","46"]

    # --- time‑domain plot ----------------------------------------------------
    for c in range(C):
        axes[0].plot(trig_np[:,c], label=f"ch {ch_names[c]}")
    axes[0].set_title(f"Estimated 75‑sample Trigger – model {model_id}")
    axes[0].set_xlabel("sample"); axes[0].set_ylabel("amplitude")
    axes[0].legend(); axes[0].grid(alpha=.3)

    # --- frequency‑domain (magnitude of Fourier coeffs) ----------------------
    fft = np.abs(np.fft.rfft(trig_np, axis=0))
    freqs = np.fft.rfftfreq(H, d=1/H)
    for c in range(C):
        axes[1].stem(freqs, fft[:,c], linefmt=f"C{c}-", markerfmt=f"C{c}o", basefmt=" ")
    axes[1].set_xlim(0, 0.15); axes[1].set_xlabel("normalised freq")
    axes[1].set_ylabel("|coeff|"); axes[1].grid(alpha=.3)

    fig.tight_layout()
    png_path = out_dir / f"trigger_{model_id:02d}.png"
    fig.savefig(png_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"  ↳ saved trigger visualisation → {png_path}")
# ---------------------------------------------------------------------------
