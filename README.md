# Accessory CNN — Real-Time Glasses / Hat / No-Accessory Classifier

A lightweight PyTorch model that recognises whether a person on camera is wearing **glasses**, a **hat**, or **no accessory**.  
With 600 real photos per class and ×2 augmentations, it reaches **99.63 % test accuracy** while running live at ~25 FPS on a laptop webcam.

---

## Folder structure
MidProject/   
│ ├── data/ ← train / val / test image folders (git-ignored)    
│ │ ├── train/   
│ │ ├── val/   
│ │ └── test/   
│ ├── saved_models/   
│ │ └── best_model.pth   
├── AccessoryCNN.ipynb ← single, self-contained notebook   
├── requirements.txt └── README.md  

*(No extra *src/* or *notebooks/* folders — everything lives in the notebook.)*


---

## Key choices

| Component        | Setting / Rationale |
|------------------|---------------------|
| **Input size**   | 128 × 128 RGB — small enough for real-time, large enough to see accessories |
| **Dataset**      | 600 raw photos × 3 classes → 70 / 15 / 15 split → **train** set tripled via augmentation (1 260 images) |
| **Augmentations**| Random crop, flip, colour-jitter; generated *after* the split to avoid leakage |
| **Model**        | 3-block CNN with **Dropout 0.5** (“slow-and-steady” regularisation) |
| **Optimiser**    | Adam (lr = 5 × 10⁻⁴) |
| **Epochs**       | 8 — enough to converge, no over-fitting |
| **Metrics**      | Val Acc = 0.9926 • **Test Acc = 0.9963** |

> **Slow and steady** = training just long enough to converge, with dropout and moderate augmentations, rather than aggressive LR tricks.

---


## Results

| Split       | Accuracy | Loss |
|-------------|---------:|-----:|
| Validation  | 0.9926   | 0.3438 |
| **Test**    | **0.9963** | 0.3100 |

---


## Quick start

```bash
git clone https://github.com/YehonatanRavoach/AccessoryCNN-Project.git
cd AccessoryCNN-Project
python -m venv .venv && .\\.venv\\Scripts\\activate      # Windows venv
pip install -r requirements.txt
jupyter notebook AccessoryCNN.ipynb                      # run everything
