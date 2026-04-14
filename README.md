# HetuNet: CNN-based Protein Reconstruction

A modular deep learning pipeline for spatio-temporal protein reconstruction using multi-scale CNN architecture.

## Project Structure

```
HetuNet/
├── main.py                           # Main entry point
├── requirements.txt                  # Project dependencies
├── README.md                         # This file
├── .gitignore                        # Git ignore rules
├── src/                              # Source code directory
│   ├── __init__.py                   # Package initializer
│   ├── model.py                      # CNN model definitions
│   ├── dataset.py                    # Dataset and DataLoader classes
│   ├── data_loader.py                # Data preprocessing utilities
│   ├── train.py                      # Training logic
│   ├── config.py                     # Argument parser
│   └── utils.py                      # Utility functions
└── demo/                             # Demo data
    ├── demo_IF_image.tif             # Demo input image
    ├── demo_MS_data.csv              # Demo input protein_data
    └── demo_mask.pkl                 # Demo input mask
```

## Features

- **Multi-scale CNN Architecture**: Extracts features at different scales for robust prediction
- **Gated Protein Head**: Adaptive feature mixing with attention mechanism
- **Spatio-temporal Learning**: Handles both row and column protein expression patterns
- **Correlation Loss**: Incorporates Pearson correlation for better prediction alignment
- **Checkpoint Resume**: Automatically resumes from the latest checkpoint
- **Multi-GPU Support**: PyTorch's built-in multi-GPU training capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Wandershy/HetuNet.git
cd HetuNet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python main.py \
    --image_path /path/to/image.ome.tif \
    --mask_path /path/to/mask.pkl \
    --protein_path /path/to/protein_data.csv \
    --output_dir ./outputs
```

### Advanced Options

```bash
python main.py \
    --image_path /path/to/image.ome.tif \
    --mask_path /path/to/mask.pkl \
    --protein_path /path/to/protein_data.csv \
    --output_dir ./outputs \
    --epochs 150 \
    --batch_size 16 \
    --lr 0.0001 \
    --tv_lambda 1e-5 \
    --correlation_lambda 1.0 \
    --patch_size 25 \
    --seed 888 \
    --num_workers 4 \
    --protein_name "VIM,CDH1"
```

### Training from Scratch

To ignore existing checkpoints and start fresh:
```bash
python main.py \
    --image_path /path/to/image.ome.tif \
    --mask_path /path/to/mask.pkl \
    --protein_path /path/to/protein_data.csv \
    --output_dir ./outputs \
    --no_resume
```

## Command-line Arguments

### Required Arguments

- `--image_path`: Path to the high-resolution ome.tif image file
- `--mask_path`: Path to the tissue mask pickle file
- `--protein_path`: Path to the protein data CSV file
- `--output_dir`: Directory to save checkpoints and logs

### Optional Arguments

- `--epochs`: Number of training epochs (default: 150)
- `--batch_size`: Batch size for training (default: 16)
- `--lr`: Base learning rate (default: 0.0001)
- `--cnn_lr_fold`: CNN learning rate multiplier (default: 1.0)
- `--tv_lambda`: Total Variation regularization weight (default: 1e-5)
- `--correlation_lambda`: Pearson correlation loss weight (default: 1.0)
- `--patch_size`: Size of square patches (default: 25)
- `--seed`: Random seed for reproducibility (default: 888)
- `--num_workers`: Number of data loading workers (default: 4)
- `--no_resume`: Start training from scratch, ignoring checkpoints
- `--protein_name`: Specific protein(s) to train (default: "all")

## Output

The training process generates the following outputs in the specified output directory:

- `epoch_XXXX.pth`: Checkpoint files containing model weights, optimizer state, and predictions
- `loss_curve_total.png`: Total training loss curve
- `loss_curve_mae.png`: MAE reconstruction loss curve
- `loss_curve_tv.png`: TV regularization loss curve

## Model Architecture

### MultiScalePatchCNN
- Multi-stage convolutional network
- Outputs features at shallow, middle, and deep scales
- Uses SiLU activation and batch normalization

### GatedProteinHead
- Learnable gating mechanism for multi-scale feature fusion
- Squeeze-and-Excitation (SE) attention
- Protein-specific prediction head

### Loss Functions
- L1 (MAE) Loss: Reconstruction accuracy
- Total Variation Loss: Spatial smoothness
- Pearson Correlation Loss: Prediction-target alignment

## Data Format

### Image File
- Format: OME-TIFF
- Multi-channel high-resolution microscopy image
- Expected to have multiple channels that will be preprocessed

### Mask File
- Format: Pickle (.pkl)
- Boolean mask indicating tissue regions
- Shape should match the low-resolution grid

### Protein Data File
- Format: CSV
- Columns: 'Genes', 'Names', followed by row and column expression values
- Each protein has H+W values (H rows, W columns)

## License

Please refer to the repository license file.

## Citation

If you use this code in your research, please cite our work.

## Contact

For questions and support, please open an issue on GitHub or contact the repository owner.
