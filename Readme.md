# Motion Planning Predictor

## Project Overview

This project implements a deep learning model to predict robot motion planning feasibility. The model determines whether a robot can move from a start position to a goal position while avoiding obstacles, and if infeasible, identifies which robot link (joint segment) first collides with obstacles.

## What It Does

The system solves a multi-class classification problem with 5 classes:
- Feasible: Path is collision-free
- Link 1: First joint fails first
- Link 2: Second joint fails first  
- Link 3: Third joint fails first
- Link 4: Fourth joint fails first

## Model Architecture

The project uses a Quotient-Space Transformer with three main components:

1. Obstacle Encoder: Processes 3D obstacle information (position and radius)
2. State Encoder: Encodes robot configuration (start position, goal position, degrees of freedom)
3. Classifier: Combines both encodings to predict feasibility class

Key specifications:
- Model parameters: 111,301
- Embedding dimension: 64
- Attention heads: 4
- Transformer layers: 2

## Performance

On the test dataset of 100,000 samples:
- Overall accuracy: 98.86%
- Per-class F1 scores range from 0.92-0.99
- Robust across different numbers of obstacles and degrees of freedom

## Project Files

- model.py: Defines the Quotient-Space Transformer and dataset loader
- training.py: Training script with loss computation and model optimization
- evaluate.py: Evaluation script that generates metrics and visualizations
- generate_dataset.py: Creates the synthetic dataset of motion planning scenarios
- generate_figures.py: Visualizes training results and evaluation metrics
- dataset.json: 100,000 training samples with robot configurations and obstacles
- best_model_weights.pth: Pre-trained model weights (best validation performance)
- final_model_weights.pth: Final trained model weights

## How to Run

### 1. Install Dependencies

```bash
pip install torch numpy matplotlib scikit-learn
```

### 2. Generate Dataset (Optional)

If you need to create a new dataset:

```bash
python generate_dataset.py
```

This generates dataset.json with 100,000 training samples.

### 3. Train the Model

```bash
python training.py
```

Training configuration:
- Learning rate: 0.001
- Batch size: 32
- Epochs: 52
- Optimizer: AdamW
- Loss: Cross-entropy with class weights for imbalance handling
- Learning rate schedule: Reduces learning rate on validation loss plateau

The script automatically saves:
- best_model_weights.pth (best performing model)
- final_model_weights.pth (final model)
- Training/validation loss and accuracy curves

### 4. Evaluate the Model

```bash
python evaluate.py
```

This loads the best model and generates:
- Confusion matrices (raw and normalized)
- ROC curves for each class
- Calibration diagrams
- Confidence distribution histogram
- Detailed evaluation report (eval_results/evaluation_report.txt)

### 5. Visualize Results

```bash
python generate_figures.py
```

Generates visualization plots of training convergence and model performance.

## Dataset Format

Each sample in dataset.json contains:
- start_config: Robot start position (4 values)
- goal_config: Robot goal position (4 values)
- dof: Degrees of freedom (1-4)
- obstacles: List of obstacles with position [x,y,z] and radius
- infeasibility_link: Label (0=Feasible, 1-4=which link fails)

## Usage Example

```python
import torch
from model import QuotientTransformer, QuotientSpaceDataset

# Load dataset
dataset = QuotientSpaceDataset('dataset.json')

# Create model
model = QuotientTransformer(d_model=64, n_heads=4, num_layers=2)

# Load trained weights
model.load_state_dict(torch.load('best_model_weights.pth'))
model.eval()

# Make prediction
state = torch.randn(1, 9)  # [start(4), goal(4), dof(1)]
obstacles = torch.randn(1, 10, 4)  # [x,y,z,radius] for up to 10 obstacles
padding_mask = torch.ones(1, 10, dtype=torch.bool)

with torch.no_grad():
    logits = model(state, obstacles, padding_mask)
    probabilities = torch.softmax(logits, dim=1)
    prediction = probabilities.argmax(dim=1)
```

## Results Summary

The model achieves excellent performance across all classes with balanced precision and recall. The evaluation includes:

- Absolute confusion matrix showing raw prediction counts
- Row-normalized confusion matrix showing per-class accuracy
- Per-DOF (degrees of freedom) accuracy breakdown
- Per-obstacle-count accuracy analysis
- ROC curves with area under curve (AUC) close to 1.0 for all classes
- Calibration analysis showing prediction reliability

## Key Insights

1. The model handles class imbalance well (Feasible is 61.9% of data)
2. Performance is robust across different numbers of obstacles (3-10)
3. Slightly better accuracy for robots with fewer degrees of freedom
4. High confidence predictions correlate with high accuracy
5. Calibration curves show model is well-calibrated for uncertainty estimation

## Technologies Used

- PyTorch for neural network implementation
- scikit-learn for evaluation metrics
- matplotlib for visualization
- NumPy for numerical operations
- JSON for dataset storage

## Notes

- The model runs on GPU if available (CUDA-enabled device recommended for faster training)
- Training takes approximately 2-3 hours on modern GPU hardware
- The dataset is synthetically generated with realistic motion planning scenarios
- The Quotient-Space representation effectively captures robot and obstacle geometry

---

For detailed evaluation metrics, refer to eval_results/evaluation_report.txt
