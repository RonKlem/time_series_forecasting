Forecasting pipeline

This repository contains a reproducible example pipeline to forecast monthly customer volumes.

Files:
- `data_generator.py`: creates synthetic dataset matching user's schema.
- `src/data_prep.py`: preprocessing utilities.
- `src/models.py`: baseline and gradient boosting models.
- `src/pytorch_model.py`: PyTorch LSTM/GRU model with embeddings.
- `train.py`: orchestration script to run preprocessing, train models, evaluate, and plot.

Run:
1. Create a virtualenv and install requirements from `requirements.txt`.
2. Run `python train.py` to execute the pipeline on synthetic data.

Notes:
- The script is written for clarity, not maximum speed. It trains small models for demonstration.
- If packages like LightGBM/XGBoost or torch are missing, the script will skip those models.
