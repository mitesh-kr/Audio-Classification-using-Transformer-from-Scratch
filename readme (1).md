# Audio Classification with CNN and Transformer Models

This repository contains code for audio classification models using CNN and hybrid CNN-Transformer architectures, trained on the ESC-10 dataset.

## Project Structure

- `main.py`: Main script to run training and evaluation
- `data_module.py`: Data handling classes for loading and preprocessing audio data
- `models.py`: Model architectures (CNN and CNN-Transformer)
- `trainer.py`: Training and evaluation utilities
- `requirements.txt`: Required dependencies

## Models

The repository implements the following models:

1. **CNN Model**: A convolutional neural network for audio classification
2. **CNN-Transformer Models**: Hybrid models combining CNN feature extraction with transformer encoders
   - With 1 attention head
   - With 2 attention heads
   - With 4 attention heads

## Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Dataset

The code is designed to work with the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50), specifically the ESC-10 subset. Download the dataset and extract it to a directory of your choice.

### Training

To train a model, run:

```bash
python main.py --model_type cnn --data_path /path/to/dataset
```

Available model types:
- `cnn`: CNN model
- `transformer_1_head`: CNN-Transformer with 1 attention head
- `transformer_2_head`: CNN-Transformer with 2 attention heads  
- `transformer_4_head`: CNN-Transformer with 4 attention heads

### Additional Parameters

```
--esc10                 Use ESC-10 subset (default: True)
--validation_fold       Fold to use for validation (2-5)
--batch_size            Batch size for training
--epochs                Number of epochs to train
--learning_rate         Learning rate for optimizer
--dropout               Dropout rate for transformer models
--use_wandb             Enable logging with Weights & Biases
--wandb_key             WandB API key
--project_name          WandB project name
--save_model            Save the trained model
```

## Evaluation

The training process includes k-fold cross-validation. After training, the model will be evaluated on the test set and the following metrics will be computed:

- Accuracy
- F1 score
- Confusion matrix
- ROC curves

Results are saved as PNG files and also logged to Weights & Biases if enabled.

## Example Commands

Train a CNN model:
```bash
python main.py --model_type cnn --epochs 100 --batch_size 40 --learning_rate 0.0001 --save_model
```

Train a CNN-Transformer model with 2 attention heads:
```bash
python main.py --model_type transformer_2_head --epochs 100 --batch_size 40 --learning_rate 0.001 --dropout 0.2 --save_model
```

## Results

### Performance Comparison

| Model | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Test Loss | Test Accuracy | F1 Score |
|-------|--------------|-------------------|-----------------|---------------------|-----------|---------------|----------|
| CNN Model | 0.5447 | 97.71% | 0.4004 | 85.31% | 2.4624 | 47.50% | 0.4066 |
| Transformer (1 head) | 0.7409 | 97.19% | 0.2126 | 92.81% | 1.6462 | 55.00% | 0.5187 |
| Transformer (2 head) | 0.1537 | 99.69% | 0.0208 | 99.06% | 2.8367 | 47.50% | 0.4371 |
| Transformer (4 head) | 0.2017 | 98.85% | 0.0468 | 98.75% | 2.8476 | 45.00% | 0.4252 |

### Key Observations

1. **Validation Performance**: The transformer models, particularly the 2-head variant, achieved excellent validation accuracy (99.06%), significantly outperforming the CNN model (85.31%).

2. **Test Performance**: Despite strong training and validation performance, all models showed signs of overfitting when evaluated on the test set. The 1-head transformer achieved the best test accuracy at 55%.

3. **F1 Score**: The 1-head transformer model achieved the highest F1 score (0.5187), indicating better overall classification performance across classes.

4. **Overfitting**: The large gap between validation and test performance suggests that all models, especially the multi-head transformers, are overfitting to the training data.

### Parameter Analysis

* **CNN Model:**
  * Trainable Parameters: **484,316**
  * Non-Trainable Parameters: **0**

* **Transformer Models:**
  * Trainable Parameters: **6,322,984**
  * Non-Trainable Parameters: **0**

The transformer models have approximately 13 times more parameters than the CNN model, which contributes to their capacity for learning complex patterns but also increases the risk of overfitting.

The model performance can be visualized through:
- Training and validation loss/accuracy curves
- Confusion matrix
- ROC curves

## License

This project is released under the MIT License.
