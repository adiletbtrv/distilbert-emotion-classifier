# distilbert-emotion-classifier

Fine-tuning "distilbert-base-uncased" on a 2000-sample subset of the "dair-ai/emotion" dataset for 6-class text emotion classification.

## Stack
- Python, PyTorch
- Hugging Face "transformers" and "datasets"
- "scikit-learn" for evaluation metrics

## Dataset and Training Setup
- Subset: 1600 train / 400 test samples ("seed=42")
- Labels: sadness, joy, love, anger, fear, surprise
- Epochs: 2
- Batch size: 32
- Learning rate: 3e-5, weight decay: 0.01

## Results
- Accuracy: 0.8450
- Macro F1: 0.8120

## Usage
Open "emotion_finetuning.ipynb" in Jupyter or VS Code and run all cells sequentially.
