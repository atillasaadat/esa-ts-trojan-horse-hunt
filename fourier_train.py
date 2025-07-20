# trigger_finder.py  (verbose, bug-fixed)
import math, glob, time
from pathlib import Path
from darts.models import NHiTSModel
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from IPython import embed
from visualise_trigger import visualise_trigger
#from viz import make_viz

# ------------------------------- CONFIG --------------------------------------
DATA_CSV            = "./data/clean_train_data.csv"
MODELS_DIR          = "./data/poisoned_models"            # poisoned_xx/*.pt
CLEAN_ROOT          = "./data/clean_model/clean_model.pt"    # no suffix
TEMPLATE_CSV        = "./data/sample_submission_solution.csv"
OUT_CSV             = "submission_fourier.csv"

SHOW_FIG = False          # set False on headless machines

CONTEXT   = 400           # 400 × 3 = 1200 matches checkpoint
HORIZON   = 75            # trigger & forecast length
CHANNELS  = ["channel_44", "channel_45", "channel_46"]

N_HARMONICS = 10          # Fourier terms
EPOCHS      = 2
BATCH       = 32
LR          = 2e-2
ALPHA       = 1.0         # divergence weight
LMBDA       = 1e-3        # L2 regulariser
SEED        = 1337
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
# -----------------------------------------------------------------------------

def flatten_for_nhits(x: torch.Tensor) -> torch.Tensor:
    """
    Convert [B, T, C]  →  [B, C*T, 1]  (channel‑major order).
    """
    return x.permute(0, 2, 1).reshape(x.size(0), -1, 1)

# ------------------------------ DATA -----------------------------------------
def make_windows(df, ctx, stride=HORIZON):
    x   = df[CHANNELS].values.astype(np.float32)
    idx = np.arange(0, len(x) - ctx - HORIZON, stride)
    windows = np.stack([x[i:i+ctx] for i in idx])
    return torch.from_numpy(windows)    

df       = pd.read_csv(DATA_CSV)
windows = make_windows(df, CONTEXT)
assert len(windows) > 0, (
    f"Clean file too short ({len(df)} rows). "
    "Need at least CONTEXT+HORIZON."
)
dataset  = TensorDataset(windows)
loader   = DataLoader(dataset, BATCH, shuffle=True)
print(f"Dataset windows: {len(dataset)}  |  batches/epoch: {len(loader)}")

# ------------------- FOURIER PARAMETERISATION --------------------------------
class FourierTrigger(nn.Module):
    r"""
    δ_c(t) = Σ_{k=1..N} a_{ck} cos(2πkt/T) + b_{ck} sin(2πkt/T)
    returns shape (HORIZON, n_channels)
    """
    def __init__(self, n_harm, horizon, n_ch):
        super().__init__()
        self.register_buffer("t", torch.arange(horizon).float() / horizon)  # [H]
        self.register_buffer("k", torch.arange(1, n_harm + 1).float())      # [K]

        # small random init avoids symmetry & speeds-up learning
        scale = 1e-2
        self.a = nn.Parameter(scale * torch.randn(n_ch, n_harm))
        self.b = nn.Parameter(scale * torch.randn(n_ch, n_harm))

    def forward(self):                       # (H, C)
        cos = torch.cos(2*math.pi* self.k[:,None] * self.t[None,:])  # [K,H]
        sin = torch.sin(2*math.pi* self.k[:,None] * self.t[None,:])
        trig = (self.a @ cos + self.b @ sin)                         # [C,H]
        return trig.T

# ------------------- OPTIMISATION PER POISONED MODEL -------------------------
def optimise_trigger(poisoned_lm, clean_lm):
    trig = FourierTrigger(N_HARMONICS, HORIZON, len(CHANNELS)).to(DEVICE)
    opt  = optim.Adam(trig.parameters(), LR)

    start = time.time()
    for epoch in range(1, EPOCHS+1):
        epoch_div, epoch_reg, epoch_loss = 0., 0., 0.
        for (ctx,) in loader:
            ctx = ctx.to(DEVICE)                          # [B, ctx, C]

            # assemble inputs -------------------------------------------------
            δ = trig().unsqueeze(0).to(DEVICE)            # [1,H,C]
            # Inject trigger into the *last HORIZON* timesteps of each channel
            ctx_poisoned = ctx.clone()
            ctx_poisoned[:, -HORIZON:, :] += δ          # [B,T,C]

            poisoned_in = flatten_for_nhits(ctx_poisoned)  # [B,1200,1]
            clean_in    = flatten_for_nhits(ctx)           # no trigger

            y_p = poisoned_lm((poisoned_in, None))
            y_c = clean_lm((clean_in,    None))

            diff = (y_p - y_c).abs().mean()
            reg  = torch.cat([p.view(-1) for p in trig.parameters()]).pow(2).mean()
            loss = -ALPHA*diff + LMBDA*reg

            opt.zero_grad(); loss.backward(); opt.step()

            epoch_div  += diff.item() * ctx.size(0)
            epoch_reg  += reg.item()  * ctx.size(0)
            epoch_loss += loss.item() * ctx.size(0)

        # ---- logging ----
        if epoch % 1 == 0 or epoch == 1:
            n = len(loader.dataset)
            print(f"epoch {epoch:3d}/{EPOCHS}  "
                f"loss={epoch_loss/n:+9.3f}  "
                f"Δ={epoch_div/n:7.2f}  L2={epoch_reg/n:9.2f}")

    return trig().detach().cpu().numpy()                   # (75,3)

# ------------------------------ MAIN -----------------------------------------
def load_nhits(root):
    wrap = NHiTSModel.load(
        root,
        pl_trainer_kwargs={"accelerator": "gpu" if DEVICE == "cuda" else "cpu"}
    )
    lm = wrap.model.to(DEVICE).eval()    # inner LightningModule for optimisation
    return wrap, lm

def main():
    print("Loading clean reference model …")
    clean_wrap, clean_lm = load_nhits(CLEAN_ROOT)
    #clean_lm = load_nhits(CLEAN_ROOT.rstrip(".pt"))  # strip suffix

    submit = pd.read_csv(TEMPLATE_CSV)

    poisoned_paths = sorted(
        p for p in Path(MODELS_DIR).rglob("*.pt")
        if p.stem.startswith("poisoned_")
    )
    print(f"Found {len(poisoned_paths)} poisoned checkpoints under {MODELS_DIR}")
    
    for model_file in poisoned_paths:
        model_id = int(model_file.parent.stem.split("_")[-1])
        print(f"\n==== Optimising trigger for poisoned model {model_id} ====")

        # = str(model_file.with_suffix(""))   # strip .pt
        poisoned_wrap, poisoned_lm   = load_nhits(str(model_file))


        trig = optimise_trigger(poisoned_lm, clean_lm)    # (75,3)
        visualise_trigger(trig, model_id,
                 out_dir=Path("trigger_plots"),
                 show=SHOW_FIG)

        # make_viz(poisoned_model=poisoned_wrap,
        #         clean_df=df,
        #         trigger_np=trig,
        #         model_id=model_id,
        #         show=SHOW_FIG)
        # fill submission row -------------------------------------------------
        flat = trig.T.reshape(-1)                         # (3*75,)
        submit.loc[submit.model_id == model_id, submit.columns[1:]] = flat
        #print("submission row filled: ", submit.loc[submit.model_id == model_id].values)

        # quick NMAE proxy (relative energy) just for sanity-check
        nmae = np.abs(flat).mean() / max(1e-9, np.abs(trig).max())
        print(f"  → filled submission row, NMAE proxy={nmae:.4f}")

    submit.to_csv(OUT_CSV, index=False)
    print(f"\nSaved submission to {OUT_CSV}")

if __name__ == "__main__":
    main()
