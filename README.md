# Price Predictor — SSH Workflow

## Project Structure

```
price_predictor/
├── config.py                   # ← All paths & hyperparams live here
├── requirements.txt
├── run.sh                      # ← Master launcher (read this!)
│
├── steps/
│   ├── _model.py               # Shared model class
│   ├── 01_extract_features.py  # GLiNER NER + IPQ + target encoding
│   ├── 02_text_embeddings.py   # BGE-M3 text embeddings
│   ├── 03_image_embeddings.py  # SigLIP image embeddings (with download logs)
│   ├── 04_train.py             # Multimodal MLP training
│   └── 05_predict.py           # Inference + submission CSV
│
├── utils/
│   └── logger.py               # Shared logger (stdout + file)
│
├── data/                       # Put train.csv / test.csv here
├── embeddings/                 # Auto-created .npy files
├── checkpoints/                # best_model.pt saved here
└── logs/                       # One .out + one timestamped .log per step
```

---

## Setup

```bash
# 1. Clone / copy the project folder to your server
scp -r price_predictor/ user@server:~/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy your data files
cp train.csv test.csv price_predictor/data/
```

---

## Running Steps (nohup — safe to close terminal)

```bash
cd ~/price_predictor

# Run one step in the background
bash run.sh features    # Step 1 — NER + IPQ + encoding
bash run.sh text        # Step 2 — Text embeddings
bash run.sh images      # Step 3 — Image download + embeddings
bash run.sh train       # Step 4 — Training
bash run.sh predict     # Step 5 — Inference
```

After launching, you can **safely close the terminal**. The process keeps running.

---

## Monitoring

```bash
# Check which steps are running / done
bash run.sh status

# Watch live output of any step (Ctrl+C to stop watching, process keeps running)
bash run.sh logs images
bash run.sh logs train

# Grep for failures in image downloads
grep "FAIL\|GAVE_UP\|ERROR" logs/images.out

# Count successful vs failed image downloads
grep -c "✓ ok"      logs/images.out
grep -c "GAVE_UP"   logs/images.out
```

---

## Kill a running step

```bash
bash run.sh kill images
```

---

## Run all steps sequentially (blocking)

```bash
# Useful inside a screen / tmux session
bash run.sh all
```

---

## Tips for long SSH sessions

```bash
# Option A — screen (reconnect later)
screen -S pipeline
bash run.sh all
# Ctrl+A then D to detach
# screen -r pipeline to reattach

# Option B — tmux
tmux new -s pipeline
bash run.sh all
# Ctrl+B then D to detach
# tmux attach -t pipeline

# Option C — nohup per step (simplest, no reattach needed)
bash run.sh images
# Close terminal. Come back later:
bash run.sh status
bash run.sh logs images
```
