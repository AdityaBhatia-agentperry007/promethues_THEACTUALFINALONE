# %% [markdown]
# # PROMETHEUS real The Well emulator training
# Runtime: Kaggle GPU or Google Colab GPU.
# This trains a backend-loadable `.pt` checkpoint. It does not use synthetic fallback.

# %%
!pip install -q the_well huggingface_hub

# %% [markdown]
# Upload/clone the repo, then run the training script.

# %%
%cd /content/promethues
!python ml/train_well_emulator.py \
  --dataset MHD_64 \
  --base-path hf://datasets/polymathic-ai/ \
  --frame-size 128 \
  --epochs 6 \
  --max-train-batches 600 \
  --max-val-batches 100 \
  --model-width 64 \
  --seed-bank-size 24 \
  --output /content/well_mhd64_emulator.pt

# %% [markdown]
# Download `/content/well_mhd64_emulator.pt`, place it at `backend/models/well_mhd64_emulator.pt`, restart backend.
