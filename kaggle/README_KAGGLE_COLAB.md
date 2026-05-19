# Real The Well Training: Kaggle / Colab

This is the non-fake path. It trains the backend-loadable emulator checkpoint from The Well data.

## Kaggle GPU

1. Create a Kaggle notebook with GPU enabled.
2. Upload this repository as a notebook input or clone it.
3. Run:

```bash
pip install -q the_well huggingface_hub
cd /kaggle/working/promethues
python ml/train_well_emulator.py \
  --dataset MHD_64 \
  --base-path hf://datasets/polymathic-ai/ \
  --frame-size 128 \
  --epochs 6 \
  --max-train-batches 600 \
  --max-val-batches 100 \
  --model-width 64 \
  --seed-bank-size 24 \
  --output /kaggle/working/well_mhd64_emulator.pt
```

Download `/kaggle/working/well_mhd64_emulator.pt` and place it at:

```text
backend/models/well_mhd64_emulator.pt
```

Then restart the backend. `/healthz` should show `"well_emulator_loaded": true`.

## Google Colab

```python
!pip install -q the_well huggingface_hub
!git clone <YOUR_REPO_URL> /content/promethues
%cd /content/promethues
!python ml/train_well_emulator.py --dataset MHD_64 --base-path hf://datasets/polymathic-ai/ --frame-size 128 --epochs 6 --max-train-batches 600 --max-val-batches 100 --model-width 64 --seed-bank-size 24 --output /content/well_mhd64_emulator.pt
```

Download `/content/well_mhd64_emulator.pt` and copy it into `backend/models/`.

## Black-Hole Requests

Do not claim black-hole GR from `MHD_64`. For black-hole-like prompts, train a more relevant checkpoint first, for example:

```bash
python ml/train_well_emulator.py --dataset post_neutron_star_merger --base-path hf://datasets/polymathic-ai/ --output /kaggle/working/well_post_merger_emulator.pt
```

Then set:

```bash
WELL_EMULATOR_CHECKPOINT=backend/models/well_post_merger_emulator.pt
```
