# FER 2013 - Enhanced ViT Training Script
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import (
    ViTModel, ViTConfig, ViTFeatureExtractor,
    AutoImageProcessor, AutoModelForImageClassification
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm # Use standard tqdm
import random
from collections import Counter
import math
import time
import gc # Garbage collector
import warnings
import gdown # For downloading
import zipfile # For unzipping

# Ignore specific warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore", "Using the model-agnostic default `max_length`")


# --- Configuration ---
# Assumes the script is run from the root 'fer2013-vit-moe-ensemble' directory
# Or adjust paths accordingly if run from elsewhere
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Get parent dir of 'scripts'
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
CACHE_DIR = os.path.join(ROOT_DIR, 'hf_cache')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

DATA_ZIP_PATH = os.path.join(DATA_DIR, 'fer2013.csv.zip')
DATA_CSV_PATH = os.path.join(DATA_DIR, 'fer2013.csv') # Extracted CSV path

NUM_CLASSES = 7
MAIN_IMG_SIZE = 224 # Main model's expected input size
BATCH_SIZE = 32 # Adjust based on GPU memory
EPOCHS = 5 # Adjust as needed, start small
LR = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_BEST_MODEL = True # Set to True to save the best model checkpoint
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'vit_moe_attention_fer_best.pth')

# --- Ensemble Configuration ---
ENSEMBLE_MODEL_NAMES = [
    "trpakov/vit-face-expression",
    "nateraw/vit-base-patch16-224-fer",
    "samhitaambati/vit-base-patch16-224-in21k-finetuned-fer",
    "Rajaram1996/FacialEmoRecog",
]
NUM_ENSEMBLE_MODELS = len(ENSEMBLE_MODEL_NAMES)
ENSEMBLE_AGREEMENT_THRESHOLD = math.ceil(NUM_ENSEMBLE_MODELS / 2) if NUM_ENSEMBLE_MODELS < 4 else math.floor(NUM_ENSEMBLE_MODELS / 2) + 1

# For MoE (part of the main model)
NUM_EXPERTS = 4
TOP_K_EXPERTS = 2

# For Attention (part of the main model)
NUM_ATTENTION_HEADS = 8
# LATENT_DIM will be set later based on the loaded main ViT model

# --- Utility Functions ---
def download_and_extract_data(file_id, zip_path, extract_path, expected_csv):
    if os.path.exists(expected_csv):
        print(f"Dataset already exists at {expected_csv}. Skipping download and extraction.")
        return True

    print(f"Downloading dataset (File ID: {file_id})...")
    try:
        # Google Drive file ID from the shareable link
        url = f"https://drive.google.com/uc?id={file_id}"
        # Download the file to the data directory
        gdown.download(url, zip_path, quiet=False)
        print(f"Downloaded to {zip_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")
        print("Please ensure 'gdown' is installed (`pip install gdown`) and the File ID is correct.")
        return False

    if not os.path.exists(zip_path):
        print(f"Download failed, zip file not found at {zip_path}")
        return False

    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        # The zip file contains fer2013.csv directly
        extracted_file = os.path.join(extract_path, 'fer2013.csv')
        if os.path.exists(extracted_file):
             print(f"Successfully extracted to {extracted_file}")
             # Clean up the zip file after successful extraction
             # os.remove(zip_path)
             # print(f"Removed zip file: {zip_path}") # Optional cleanup
             return True
        else:
             print(f"Error: Expected file 'fer2013.csv' not found after extraction in {extract_path}.")
             return False
    except zipfile.BadZipFile:
        print(f"Error: Bad zip file at {zip_path}. Please check the download or file integrity.")
        return False
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        return False


def parse_pixels(pixel_string):
    try:
        pixels = np.array(pixel_string.split(), dtype='uint8')
        img = pixels.reshape(48, 48)
        pil_img = Image.fromarray(img).convert('L') # Start as grayscale PIL
        return pil_img
    except Exception as e:
        print(f"Error parsing pixel string: {e}")
        return Image.new('L', (48, 48)) # Return dummy black image

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

# --- 0. Download and Prepare Data ---
print("--- Ensuring Data is Available ---")
# Google Drive file ID from the shareable link
gdrive_file_id = "1FxpKWPlrDARxR_Po9Nyf34rQS9Vt0DZn"
if not download_and_extract_data(gdrive_file_id, DATA_ZIP_PATH, DATA_DIR, DATA_CSV_PATH):
    print("Failed to prepare data. Exiting.")
    sys.exit(1) # Exit if data preparation fails

# --- 1. Data Loading ---
print("\n--- Loading Data ---")
if not os.path.exists(DATA_CSV_PATH):
    raise FileNotFoundError(f"Error: Dataset not found at {DATA_CSV_PATH}. Please check the download/extraction step.")

df = pd.read_csv(DATA_CSV_PATH)
print(f"Dataset loaded: {len(df)} samples")
print(df['Usage'].value_counts())
emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# --- 2. Load Ensemble Models ---
print("\n--- Loading Ensemble Models ---")

def load_ensemble_models(model_names, device):
    models = []
    processors = []
    loaded_names = []
    for name in tqdm(model_names, desc="Loading Ensemble Models"):
        try:
            print(f"Loading {name}...")
            processor = AutoImageProcessor.from_pretrained(name, cache_dir=CACHE_DIR)
            model = AutoModelForImageClassification.from_pretrained(
                name,
                num_labels=NUM_CLASSES, # Assume models are fine-tuned or specify ignore_mismatched_sizes=True
                ignore_mismatched_sizes=True, # Important if loading ImageNet models directly
                cache_dir=CACHE_DIR
            )
            model.eval()
            model.to(device)
            models.append(model)
            processors.append(processor)
            loaded_names.append(name)
            print(f"Loaded {name} successfully.")
            clear_gpu_memory() # Clear cache after loading each model
        except Exception as e:
            print(f"Warning: Failed to load ensemble model {name}. Error: {e}. Skipping.")
            clear_gpu_memory()
    if not models:
        raise RuntimeError("Could not load any ensemble models. Check model names, connectivity, and cache directory.")
    print(f"Successfully loaded {len(models)} ensemble models: {loaded_names}")
    return models, processors, loaded_names

# Load the actual ensemble models
ensemble_models, ensemble_processors, ensemble_model_names = load_ensemble_models(ENSEMBLE_MODEL_NAMES, DEVICE)
# Update actual number of models loaded
NUM_ENSEMBLE_MODELS = len(ensemble_models)
if NUM_ENSEMBLE_MODELS == 0:
    print("Error: No ensemble models were loaded. Cannot proceed with refinement.")
    sys.exit(1)
ENSEMBLE_AGREEMENT_THRESHOLD = math.ceil(NUM_ENSEMBLE_MODELS / 2) if NUM_ENSEMBLE_MODELS < 4 else math.floor(NUM_ENSEMBLE_MODELS / 2) + 1
print(f"Using {NUM_ENSEMBLE_MODELS} models for ensemble voting. Threshold: {ENSEMBLE_AGREEMENT_THRESHOLD}")


# --- 3. Ensemble Inference & Refinement ---
print("\n--- Refining ALL Data with Actual Ensemble Voting ---")
print("--- This may take a significant amount of time! ---")

# Dataset specifically for ensemble inference (returns PIL image for flexibility)
class EnsembleInferenceDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Return PIL Image (RGB) and original label
        img = parse_pixels(row['pixels']).convert('RGB')
        original_label = row['emotion']
        return img, original_label

# Create DataLoader for ensemble inference
# Use a smaller batch size for inference if GPU memory is limited
inference_batch_size = BATCH_SIZE // 2 if DEVICE.type == 'cuda' else BATCH_SIZE
inference_dataset = EnsembleInferenceDataset(df)
# Note: DataLoader with PIL Images might be slow without a custom collate_fn,
# but processing inside the loop gives flexibility for different processors.
inference_loader = DataLoader(inference_dataset, batch_size=inference_batch_size, shuffle=False, num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False)

all_original_labels = df['emotion'].tolist() # Get all original labels directly
mismatch_count = 0

inference_start_time = time.time()

# Run inference with each model
model_predictions = [[] for _ in range(NUM_ENSEMBLE_MODELS)]

with torch.no_grad():
    for model_idx, model in enumerate(ensemble_models):
        print(f"\nRunning inference with Model {model_idx+1}/{NUM_ENSEMBLE_MODELS} ({ensemble_model_names[model_idx]})")
        processor = ensemble_processors[model_idx] # Get the processor for this model
        current_model_preds = []

        for batch_pil_images, _ in tqdm(inference_loader, desc=f"Model {model_idx+1} Inference"):
            # Process the batch of PIL images using the *current* model's processor
            try:
                inputs = processor(images=list(batch_pil_images), return_tensors="pt", padding=True, truncation=True)
                pixel_values = inputs['pixel_values'].to(DEVICE)

                outputs = model(pixel_values)
                predictions = torch.argmax(outputs.logits, dim=1)
                current_model_preds.extend(predictions.cpu().tolist())
                clear_gpu_memory() # Clear per batch if memory is tight

            except Exception as e:
                print(f"Error during batch processing for model {model_idx+1}: {e}")
                # Add placeholder predictions (e.g., -1 or a default) if an error occurs
                current_model_preds.extend([-1] * len(batch_pil_images)) # Use -1 to indicate error

        # Store predictions for this model, ensuring length matches dataset
        if len(current_model_preds) == len(df):
            model_predictions[model_idx] = current_model_preds
        else:
            print(f"Warning: Length mismatch for model {model_idx+1} predictions ({len(current_model_preds)}) vs dataset ({len(df)}). Check for errors.")
            # Handle mismatch: Pad with -1 or attempt to fix logic
            model_predictions[model_idx] = (current_model_preds + [-1] * len(df))[:len(df)] # Pad/truncate defensively


        clear_gpu_memory() # Clear after each model's full run

# Perform Voting
print("\nPerforming ensemble voting...")
final_refined_labels = []
for i in tqdm(range(len(df)), desc="Voting"):
    votes = []
    for model_idx in range(NUM_ENSEMBLE_MODELS):
         # Check if prediction exists and is valid (not -1 from error handling)
         if i < len(model_predictions[model_idx]) and model_predictions[model_idx][i] != -1:
              votes.append(model_predictions[model_idx][i])

    original_label = all_original_labels[i] # Get corresponding original label

    if not votes: # No valid votes for this sample
        final_refined_labels.append(original_label)
        continue

    vote_counts = Counter(votes)
    most_common = vote_counts.most_common(1)

    if most_common:
        majority_vote_label, majority_count = most_common[0]
        # Check if there's a qualifying majority based on *valid* votes
        if majority_count >= ENSEMBLE_AGREEMENT_THRESHOLD:
            final_refined_labels.append(majority_vote_label)
            if majority_vote_label != original_label:
                mismatch_count += 1
        else:
            # No clear majority, keep original label
            final_refined_labels.append(original_label)
    else:
         # Should not happen if votes list is not empty, but handle defensively
         final_refined_labels.append(original_label)

inference_duration = time.time() - inference_start_time
print(f"Ensemble inference and voting took {inference_duration // 60:.0f}m {inference_duration % 60:.0f}s.")

# Add refined labels to the dataframe
df['refined_emotion'] = final_refined_labels
print(f"Refined labels generated for {len(final_refined_labels)} samples.")
print(f"Relabeled {mismatch_count} samples across the entire dataset based on ensemble majority.")
print(f"Original vs Refined counts (first 10):")
print(df[['emotion', 'refined_emotion']].head(10))

# Optional: Analyze label changes per split
print("\nLabel change statistics per split:")
for usage in df['Usage'].unique():
    split_df = df[df['Usage'] == usage]
    if not split_df.empty:
        changes = (split_df['emotion'] != split_df['refined_emotion']).sum()
        print(f"  {usage}: {changes} / {len(split_df)} labels changed.")
    else:
        print(f"  {usage}: 0 / 0 samples.")


# Release ensemble models from memory
print("Releasing ensemble models from memory...")
del ensemble_models, ensemble_processors, inference_dataset, inference_loader, model_predictions
clear_gpu_memory()


# --- 4. Main Model Definition (ViT + MoE + Attention) ---
print("\n--- Defining Main Enhanced ViT Model ---")

# Load base ViT for the main model (e.g., the standard google/vit one)
main_vit_model_name = 'google/vit-base-patch16-224-in21k'
try:
    main_feature_extractor = ViTFeatureExtractor.from_pretrained(main_vit_model_name, cache_dir=CACHE_DIR)
    # Ensure main feature extractor uses the target size
    if main_feature_extractor.size != MAIN_IMG_SIZE: # Check if size is int or dict
        print(f"Adjusting Main ViT FE size config to {MAIN_IMG_SIZE}.")
        # Handle both dict and int size representations
        if isinstance(main_feature_extractor.size, int):
            main_feature_extractor.size = MAIN_IMG_SIZE
        elif isinstance(main_feature_extractor.size, dict):
             main_feature_extractor.size = {"height": MAIN_IMG_SIZE, "width": MAIN_IMG_SIZE}
             # Also adjust crop_size if it exists and is different
             if hasattr(main_feature_extractor, 'crop_size'):
                 main_feature_extractor.crop_size = {"height": MAIN_IMG_SIZE, "width": MAIN_IMG_SIZE}
        else:
             print("Warning: Unknown feature extractor size format. Size might not be set correctly.")

    # Get Latent Dim from the main model's config
    main_vit_config = ViTConfig.from_pretrained(main_vit_model_name, cache_dir=CACHE_DIR)
    LATENT_DIM = main_vit_config.hidden_size
    print(f"Main ViT Model Latent Dimension: {LATENT_DIM}")

except Exception as e:
    print(f"Error loading main ViT model/config '{main_vit_model_name}': {e}")
    print("Please check the model name and internet connection.")
    sys.exit(1)

# --- MoE Layer, LatentAttention, ViTMoEAttentionFER Class Definitions ---

# 5.1 MoE Layer
class MoELayer(nn.Module):
    def __init__(self, input_dim, num_experts, top_k, expert_hidden_dim=None):
        super().__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.top_k = min(top_k, num_experts) # Ensure top_k is not > num_experts

        if expert_hidden_dim is None:
            expert_hidden_dim = input_dim * 2 # A simple heuristic

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden_dim),
                nn.GELU(), # Use GELU like in ViT
                nn.Linear(expert_hidden_dim, input_dim)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        original_shape = x.shape
        is_sequence = len(original_shape) > 2
        x_flat = x.reshape(-1, self.input_dim) if is_sequence else x

        gate_logits = self.gate(x_flat)
        gate_scores = torch.softmax(gate_logits, dim=-1)
        # Handle cases where top_k=1 requires slightly different indexing
        if self.top_k == 1:
            top_k_weights, top_k_indices = torch.max(gate_scores, dim=-1)
            top_k_weights = top_k_weights.unsqueeze(-1)
            top_k_indices = top_k_indices.unsqueeze(-1)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True) # Normalize (trivial for top_k=1)
        else:
            top_k_weights, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True) # Normalize scores

        output = torch.zeros_like(x_flat)
        # flat_batch_indices = torch.arange(x_flat.size(0), device=x.device) # Not needed with mask approach

        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]
            current_weights = top_k_weights[:, i].unsqueeze(-1)
            for exp_idx in range(self.num_experts):
                mask = (expert_indices == exp_idx)
                if mask.any():
                    selected_inputs = x_flat[mask]
                    if selected_inputs.numel() > 0: # Ensure there are inputs to process
                        expert_output = self.experts[exp_idx](selected_inputs)
                        output[mask] += expert_output * current_weights[mask]

        if is_sequence:
            output = output.reshape(original_shape)
        return output

# 5.2 Multi-head Latent Attention Layer
class LatentAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, hidden_states):
        # Query, Key, Value are all the same hidden states for self-attention
        attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        output = self.norm(hidden_states + attn_output) # Add & Norm (Residual connection)
        return output

# 5.3 Combined Model
class ViTMoEAttentionFER(nn.Module):
    def __init__(self, vit_model_name, num_classes, num_experts, top_k_experts, num_attention_heads, latent_dim, cache_dir):
        super().__init__()
        print(f"Initializing main model using {vit_model_name}")
        self.vit = ViTModel.from_pretrained(vit_model_name, cache_dir=cache_dir)
        model_config = self.vit.config
        if latent_dim != model_config.hidden_size:
             print(f"Warning: Overriding LATENT_DIM from {latent_dim} to model's hidden size {model_config.hidden_size}")
             latent_dim = model_config.hidden_size
        self.latent_dim = latent_dim

        # Optional: Freeze base ViT (uncomment if needed)
        # print("Freezing base ViT model parameters.")
        # for param in self.vit.parameters():
        #     param.requires_grad = False

        # Use latent_dim consistently now it's finalized
        self.latent_attention = LatentAttention(self.latent_dim, num_attention_heads)
        self.norm_after_attention = nn.LayerNorm(self.latent_dim)
        self.moe = MoELayer(self.latent_dim, num_experts, top_k_experts)
        self.norm_after_moe = nn.LayerNorm(self.latent_dim)
        self.classifier = nn.Linear(self.latent_dim, num_classes)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        last_hidden_state = outputs.last_hidden_state # Shape: (batch_size, sequence_length, hidden_size)

        # Apply attention across the sequence dimension (including CLS token)
        attended_hidden_state = self.latent_attention(last_hidden_state)

        # Extract the CLS token representation *after* attention
        cls_token_attended = attended_hidden_state[:, 0] # Shape: (batch_size, hidden_size)

        # Apply normalization after attention (on CLS token)
        cls_token_attended_norm = self.norm_after_attention(cls_token_attended)

        # Apply MoE to the attended and normed CLS token
        moe_output = self.moe(cls_token_attended_norm) # Shape: (batch_size, hidden_size)

        # Combine CLS token representation with MoE output (residual connection)
        final_representation = cls_token_attended_norm + moe_output
        final_representation = self.norm_after_moe(final_representation) # Final normalization

        logits = self.classifier(final_representation) # Classify based on the enhanced CLS token
        return logits

# Instantiate the main model
try:
    model = ViTMoEAttentionFER(
        vit_model_name=main_vit_model_name,
        num_classes=NUM_CLASSES,
        num_experts=NUM_EXPERTS,
        top_k_experts=TOP_K_EXPERTS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        latent_dim=LATENT_DIM,
        cache_dir=CACHE_DIR
    )
    model.to(DEVICE)
    LATENT_DIM = model.latent_dim # Update global LATENT_DIM if it was changed internally

    print("Main Model created and moved to device:", DEVICE)
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
except Exception as e:
    print(f"Error creating main model: {e}")
    sys.exit(1)

# --- 5. PyTorch Dataset and DataLoader for Main Model ---
print("\n--- Creating Datasets and DataLoaders for Main Model (using refined labels) ---")

class MainFERDataset(Dataset):
    def __init__(self, df, usage, feature_extractor, augment=False):
        # Filter dataframe by usage split BEFORE reset_index
        self.df = df[df['Usage'] == usage].reset_index(drop=True)
        if len(self.df) == 0:
             print(f"Warning: No data found for usage split '{usage}'")
        self.usage = usage
        self.feature_extractor = feature_extractor
        self.augment = augment

        # Define transformations
        # Base transforms (resize, to tensor, normalize) are usually handled by feature_extractor
        self.augment_transform = None
        if self.augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2), # ViT expects RGB
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx >= len(self.df):
             raise IndexError("Index out of bounds")
        row = self.df.iloc[idx]
        img = parse_pixels(row['pixels']).convert('RGB') # Ensure RGB

        if self.augment and self.augment_transform:
             img = self.augment_transform(img)

        try:
            # Use the main model's feature extractor (handles resize, normalize, to tensor)
            inputs = self.feature_extractor(images=img, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0) # Remove batch dim
        except Exception as e:
             print(f"Error processing image at index {idx} ({self.usage}) with Main FeatureExtractor: {e}")
             # Try to get size dynamically
             img_size = MAIN_IMG_SIZE # Fallback
             if isinstance(self.feature_extractor.size, int):
                 img_size = self.feature_extractor.size
             elif isinstance(self.feature_extractor.size, dict):
                 img_size = self.feature_extractor.size.get('height', MAIN_IMG_SIZE)
             pixel_values = torch.zeros((3, img_size, img_size)) # Return dummy tensor

        # Use the refined label column
        label = row['refined_emotion']

        return pixel_values, torch.tensor(label, dtype=torch.long)

# Create datasets using the *modified* df containing 'refined_emotion'
train_dataset = MainFERDataset(df, 'Training', main_feature_extractor, augment=True)
val_dataset = MainFERDataset(df, 'PublicTest', main_feature_extractor, augment=False)
test_dataset = MainFERDataset(df, 'PrivateTest', main_feature_extractor, augment=False)

# Create dataloaders
num_workers = min(os.cpu_count() // 2, 4) if DEVICE.type == 'cuda' else 0
# num_workers = 0 # Use 0 if persistent workers cause issues
print(f"Using {num_workers} dataloader workers.")

# Use the main BATCH_SIZE config
# Set persistent_workers based on num_workers
persistent = num_workers > 0

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers,
                          pin_memory=True if DEVICE.type == 'cuda' else False, persistent_workers=persistent)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers,
                        pin_memory=True if DEVICE.type == 'cuda' else False, persistent_workers=persistent)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers,
                         pin_memory=True if DEVICE.type == 'cuda' else False, persistent_workers=persistent)

print(f"DataLoaders created (using refined labels for all splits):")
# Handle potentially empty datasets
print(f"  Training: {len(train_dataset)} samples, {len(train_loader)} batches")
print(f"  Validation: {len(val_dataset)} samples, {len(val_loader)} batches")
print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")

# --- 6. Training Pipeline Setup ---
print("\n--- Setting up Training Pipeline ---")
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.2, verbose=True)

# --- 7. Training and Evaluation Loop ---
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False, unit="batch")
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        batch_size = labels.size(0)
        total_samples += batch_size
        correct_predictions += (predicted == labels).sum().item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{total_loss / (i + 1):.4f}", acc=f"{correct_predictions / total_samples:.4f}")

    # Handle case where dataloader is empty
    if not total_samples:
        return 0.0, 0.0

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device, split_name="Evaluating"):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    progress_bar = tqdm(dataloader, desc=split_name, leave=False, unit="batch")
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            batch_size = labels.size(0)
            total_samples += batch_size
            correct_predictions += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct_predictions / total_samples:.4f}")

    # Handle case where dataloader is empty
    if not total_samples:
         return 0.0, 0.0, [], []

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc, all_labels, all_preds

print("\n--- Starting Training on Refined Data ---")
start_time = time.time()
best_val_acc = 0.0
best_epoch = -1

# Check if datasets are empty before starting loop
if len(train_dataset) == 0 or len(val_dataset) == 0:
    print("Error: Training or Validation dataset is empty. Cannot train.")
else:
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        clear_gpu_memory()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        clear_gpu_memory()

        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE, split_name="Validation")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

        epoch_duration = time.time() - epoch_start_time
        print(f"  Epoch Duration: {epoch_duration // 60:.0f}m {epoch_duration % 60:.1f}s")

        if scheduler:
            scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            print(f"  *** New best validation accuracy: {best_val_acc:.4f} (Epoch {best_epoch}) ***")
            if SAVE_BEST_MODEL:
                try:
                    torch.save(model.state_dict(), BEST_MODEL_PATH)
                    print(f"  Best model saved to {BEST_MODEL_PATH}")
                except Exception as e:
                    print(f"  Error saving model: {e}")

    training_time = time.time() - start_time
    print(f"\n--- Training Finished in {training_time // 60:.0f}m {training_time % 60:.0f}s ---")
    print(f"Best Validation Accuracy on Refined Data: {best_val_acc:.4f} achieved at Epoch {best_epoch}")

# --- 8. Final Evaluation on the *Refined* Test Set ---
print("\n--- Evaluating on the *Refined* Test Set ---")
# Load best model if saved and exists
if SAVE_BEST_MODEL and os.path.exists(BEST_MODEL_PATH) and best_epoch != -1:
    print(f"Loading best model from {BEST_MODEL_PATH} for final evaluation...")
    try:
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading best model state_dict: {e}. Evaluating with the last epoch model.")
elif not SAVE_BEST_MODEL:
     print("Evaluating with model from the last epoch (saving was disabled).")
else: # Model saving was enabled but file doesn't exist or no improvement seen
     print("Best model file not found or no improvement seen. Evaluating with model from the last epoch.")

# Check if test dataset is empty
if len(test_dataset) == 0:
    print("Test dataset is empty. Skipping final evaluation.")
else:
    clear_gpu_memory()
    test_loss, test_acc, test_labels, test_preds = evaluate(model, test_loader, criterion, DEVICE, split_name="Test Set")

    print(f"\nRefined Test Set Loss: {test_loss:.4f}")
    print(f"Refined Test Set Accuracy: {test_acc:.4f}")

    if test_labels and test_preds: # Ensure we have results to report
        print("\nClassification Report (Refined Test Set):")
        class_names = list(emotion_map.values())
        try:
            # Use the refined labels from the dataloader (test_labels) as ground truth
            print(classification_report(test_labels, test_preds, target_names=class_names, digits=4, zero_division=0))
        except Exception as e:
            print(f"Could not generate classification report: {e}")
            print("Accuracy:", accuracy_score(test_labels, test_preds))


# --- Optional: Compare with Original Test Labels ---
print("\n--- Comparison with Original Test Labels (For Analysis Only) ---")
original_test_df = df[df['Usage'] == 'PrivateTest']
if not original_test_df.empty and test_labels and test_preds and len(original_test_df) == len(test_preds):
     original_test_labels = original_test_df['emotion'].tolist()
     print("Accuracy against ORIGINAL Test Labels:")
     print(f"{accuracy_score(original_test_labels, test_preds):.4f}")
     print("\nClassification Report against ORIGINAL Test Labels:")
     class_names = list(emotion_map.values())
     try:
         print(classification_report(original_test_labels, test_preds, target_names=class_names, digits=4, zero_division=0))
     except Exception as e:
        print(f"Could not generate original classification report: {e}")
else:
     print("Could not compare with original labels (test set empty, mismatch in lengths, or no predictions generated).")


print("\n--- Script Complete ---") 