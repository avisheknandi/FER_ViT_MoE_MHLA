# FER2013 Classification with ViT + MoE + Attention Ensemble Refinement

This project implements a Vision Transformer (ViT) based model enhanced with Mixture-of-Experts (MoE) and Latent Attention layers for facial expression recognition on the FER2013 dataset. It includes a data refinement step using an ensemble of pre-trained models before training the main enhanced ViT model.

## Features

*   Downloads and prepares the FER2013 dataset.
*   Loads multiple pre-trained facial expression models for ensemble inference.
*   Refines dataset labels based on ensemble majority voting.
*   Defines a custom ViT model incorporating:
    *   A base pre-trained ViT (`google/vit-base-patch16-224-in21k`).
    *   A Multi-Head Latent Attention layer operating on the ViT's sequence output.
    *   A Mixture-of-Experts (MoE) layer applied to the CLS token representation.
*   Trains the enhanced ViT model on the refined FER2013 data.
*   Evaluates the model on the refined test set.
*   Provides comparison with original test set labels (for analysis).

## Project Structure

```
fer2013-vit-moe-ensemble/
├── .gitignore # Specifies intentionally untracked files
├── data/ # Data directory (dataset downloaded here)
│   └── .gitkeep
├── models/ # Saved model checkpoints directory
│   └── .gitkeep
├── scripts/
│   └── train.py # Main training and evaluation script
├── requirements.txt # Project dependencies
└── README.md # This file
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/avisheknandi/FER_ViT_MoE_MHLA
    cd FER_ViT_MoE_MHLA
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Depending on your system and CUDA version, you might need specific PyTorch installation instructions. See [https://pytorch.org/](https://pytorch.org/)*

4.  **Data Download:** The `train.py` script will automatically attempt to download the FER2013 dataset using `gdown` and extract it to the `data/` directory when run for the first time. Ensure you have an internet connection.

## Running the Script

Execute the main training script:

```bash
python scripts/train.py
```

### Script Execution Steps:

1.  Checks if `data/fer2013.csv` exists. If not, downloads and extracts `fer2013.csv.zip`.
2.  Loads the specified ensemble models from Hugging Face (requires internet, downloads models to `hf_cache/`).
3.  Performs inference with each ensemble model on the entire dataset (this can take a long time).
4.  Refines the labels based on the ensemble majority vote.
5.  Initializes the main ViT+MoE+Attention model.
6.  Creates DataLoaders using the refined labels.
7.  Trains the main model for the specified number of epochs, using the refined training data and evaluating on the refined validation data.
8.  Saves the best model checkpoint (based on validation accuracy) to `models/vit_moe_attention_fer_best.pth`.
9.  Evaluates the best model on the refined test set and prints results.
10. Optionally compares test predictions against the original (unrefined) test labels.

## Configuration

Key parameters can be adjusted directly in the `scripts/train.py` file:

*   `DATA_DIR`, `MODEL_DIR`, `CACHE_DIR`: Directory paths.
*   `NUM_CLASSES`, `MAIN_IMG_SIZE`, `BATCH_SIZE`, `EPOCHS`, `LR`: Training hyperparameters.
*   `ENSEMBLE_MODEL_NAMES`: List of Hugging Face model identifiers for the ensemble.
*   `NUM_EXPERTS`, `TOP_K_EXPERTS`: MoE layer configuration.
*   `NUM_ATTENTION_HEADS`: Latent Attention configuration.
*   `SAVE_BEST_MODEL`: Boolean flag to enable/disable saving the best model.

## Notes

*   The ensemble refinement step requires significant time and computational resources (GPU recommended).
*   Ensure the Hugging Face models listed in `ENSEMBLE_MODEL_NAMES` are accessible and compatible (or set `ignore_mismatched_sizes=True`).
*   The Hugging Face cache (`hf_cache/`) and downloaded data/models are excluded by `.gitignore`.
*   This structure provides a clean, organized, and standard way to manage your machine learning project, making it ready for version control with Git and sharing on platforms like GitHub. Remember to initialize Git in the root folder (`git init`) and make your first commit (`git add .`, `git commit -m "Initial commit"`). 
