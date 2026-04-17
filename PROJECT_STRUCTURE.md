# HetuNet Project Structure

## Visual Overview

```
HetuNet/
│
├── 📄 main.py                              # Main entry point
│   └── Orchestrates: parse_args() → load_data() → train_model()
│
├── 📄 Visualization.ipynb                  # Process the reconstructed spatial protein maps for visualization and downstream analysis.
├── 📄 requirements.txt                     # Project dependencies
├── 📄 README.md                            # User documentation
├── 📄 REFACTORING_NOTES.md                 # Developer documentation
├── 📄 .gitignore                           # Git ignore rules
│
├── 📁 src/                                 # Source code package
│   │
│   ├── 📄 __init__.py                      # Package initializer
│   │
│   ├── 📄 model.py                         # Neural Network Models
│   │   ├── class PearsonCorrelationLoss    # Custom loss function
│   │   ├── class MultiScalePatchCNN        # Multi-scale CNN backbone
│   │   └── class GatedProteinHead          # Protein-specific prediction head
│   │
│   ├── 📄 dataset.py                       # Data Loading
│   │   ├── class SpatioTemporalDataset     # Training dataset
│   │   ├── class InferenceDataset          # Inference dataset
│   │   └── def custom_collate_fn()         # Custom batching logic
│   │
│   ├── 📄 data_loader.py                   # Data Preprocessing
│   │   ├── def load_high_res_image()       # Load and preprocess images
│   │   ├── def load_mask()                 # Load tissue mask
│   │   └── def load_protein_data()         # Load protein expression data
│   │
│   ├── 📄 train.py                         # Training Logic
│   │   └── def train_model_with_L1()       # Main training loop
│   │
│   ├── 📄 config.py                        # Configuration
│   │   └── def parse_args()                # Command-line argument parser
│   │
│   └── 📄 utils.py                         # Utilities
│       ├── def setup_logging()             # Logging configuration
│       ├── def set_seed()                  # Random seed setter
│       ├── def plot_loss_curve()           # Loss visualization
│       └── def fill_na_with_neighbor_mean() # Data preprocessing helper
│
└── 📁 demo/                                # Demo data
    ├── demo_IF_image.tif                   # Demo input image
    ├── demo_row_col_data.csv               # Demo input expression_data
    ├── demo_mask.pkl                       # Demo input mask
    └── demo_ground_truth.h5ad              # Demo ground truth
```

## Data Flow

```
Command Line Arguments
         ↓
    parse_args() (config.py)
         ↓
    set_seed() (utils.py)
         ↓
    ┌────────────────────────────────┐
    │  Data Loading (data_loader.py) │
    ├────────────────────────────────┤
    │  • load_high_res_image()       │
    │  • load_mask()                 │
    │  • load_protein_data()         │
    └────────────────────────────────┘
         ↓
    ┌────────────────────────────────┐
    │  Dataset Creation (dataset.py) │
    ├────────────────────────────────┤
    │  • SpatioTemporalDataset       │
    │  • DataLoader + collate_fn     │
    └────────────────────────────────┘
         ↓
    ┌────────────────────────────────┐
    │  Model Initialization          │
    │  (model.py)                    │
    ├────────────────────────────────┤
    │  • MultiScalePatchCNN          │
    │  • GatedProteinHead (per prot) │
    │  • PearsonCorrelationLoss      │
    └────────────────────────────────┘
         ↓
    ┌────────────────────────────────┐
    │  Training Loop (train.py)      │
    ├────────────────────────────────┤
    │  • Forward pass                │
    │  • Loss computation            │
    │  • Backward pass               │
    │  • Checkpoint saving           │
    │  • Visualization               │
    └────────────────────────────────┘
         ↓
    Trained Model + Predictions
```

## Module Dependencies

```
main.py
  ├── → config.py (parse_args)
  ├── → utils.py (setup_logging, set_seed)
  ├── → data_loader.py (load_high_res_image, load_mask, load_protein_data)
  └── → train.py (train_model_with_L1)
         ├── → model.py (MultiScalePatchCNN, GatedProteinHead, PearsonCorrelationLoss)
         ├── → dataset.py (SpatioTemporalDataset, InferenceDataset, custom_collate_fn)
         └── → utils.py (plot_loss_curve)

data_loader.py
  └── → utils.py (fill_na_with_neighbor_mean)
```

## Component Sizes

| Component | Lines of Code | Purpose |
|-----------|--------------|---------|
| main.py | 41 | Entry point and orchestration |
| src/model.py | 114 | Neural network architecture |
| src/dataset.py | 155 | Data loading and batching |
| src/data_loader.py | 130 | Data preprocessing |
| src/train.py | 310 | Training algorithm |
| src/config.py | 124 | Configuration management |
| src/utils.py | 75 | Utility functions |
| **Total** | **949** | **(vs. 416 in original)** |

*Note: The increase in lines is due to:*
- Better code formatting and spacing
- Comprehensive docstrings
- Module headers and imports
- Improved readability

## Key Design Principles

1. **Single Responsibility**: Each module handles one aspect
2. **Dependency Injection**: Functions receive what they need
3. **Separation of Concerns**: Clear boundaries between components
4. **Reusability**: Components can be used independently
5. **Testability**: Each module can be tested in isolation
6. **Maintainability**: Easy to locate and modify code

## Usage Example

```bash
# Basic usage - identical to original script
python main.py \
    --image_path /path/to/image.ome.tif \
    --mask_path /path/to/mask.pkl \
    --protein_path /path/to/protein_data.csv \
    --output_dir ./outputs

# Advanced usage with custom parameters
python main.py \
    --image_path /path/to/image.ome.tif \
    --mask_path /path/to/mask.pkl \
    --protein_path /path/to/protein_data.csv \
    --output_dir ./outputs \
    --epochs 200 \
    --batch_size 32 \
    --lr 0.0001 \
    --tv_lambda 1e-5 \
    --correlation_lambda 1.0 \
    --patch_size 25 \
    --protein_name "VIM,CDH1" \
    --no_resume
```

## Import Example

```python
# Using components in other scripts
from src.model import MultiScalePatchCNN, GatedProteinHead
from src.dataset import SpatioTemporalDataset
from src.utils import set_seed, setup_logging

# Setup
setup_logging()
set_seed(42)

# Create models
cnn = MultiScalePatchCNN(in_channels=4, patch_size=25)
head = GatedProteinHead()

# Use in your own training loop
# ...
```
