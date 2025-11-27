# CS 5342 Assignment 3: Automated Moderator (Bluesky Labeler)

## 1. Group Information

**Group Number 11**

**Group Members:**
- Andreas Kilbinger
- Ali Hasan
- Mark Farley
- Aagam Bakliwal

## 2. List and Description of Submitted Files

### Core Implementation Files

**`policy_proposal_labeler.py`** — Main implementation file containing the complete fraud detection labeler system. This file includes:
- **Feature Engineering**: 
  - Semantic embeddings using Sentence Transformers (384 dimensions)
  - 28 linguistic features (text statistics, URL analysis, keyword matching, etc.)
- **N-gram Model**: TF-IDF vectorization with 1-3 gram sequences, trained with XGBoost
- **Ensemble Models**: 
  - Base ensemble combining Logistic Regression, Random Forest, and XGBoost
  - Final ensemble combining base models (60%) with n-gram model (40%)
- **Training Function**: `train_model_from_csv()` - trains the complete model pipeline
- **Prediction Functions**: 
  - `predict_post(text)` - single post prediction
  - `predict_batch(texts)` - batch prediction for efficiency
- **Visualization**: `plot_model_comparison()` - generates performance comparison graphs

**`predictions.py`** — Testing script that runs predictions on predefined test cases including clear scams, clear non-scams, and ambiguous cases. Demonstrates the labeler's functionality and outputs predictions with confidence scores.

### Data Files

**`data.csv`** — Training dataset containing:
- **Post Content**: Text content of Bluesky posts
- **Ground Truth Label**: Binary labels (0 = Legitimate, 1 = Fraud)

**`test.csv`** — Test dataset used for model evaluation, containing the same structure as `data.csv` with separate test posts and their ground truth labels.

### Model Files (Generated After Training)

**`fraud_classifier.joblib`** — Main ensemble classifier model (saved after training)

**`fraud_embedder.joblib`** — Sentence transformer model for semantic embeddings (saved after training)

**`fraud_scaler.joblib`** — Feature scaler for normalizing input features (saved after training)

**`fraud_ngram_model.joblib`** — N-gram model (XGBoost classifier trained on TF-IDF features) (saved after training)

**`fraud_ngram_vectorizer.joblib`** — TF-IDF vectorizer for n-gram feature extraction (saved after training)

**`fraud_feature_selector.joblib`** — Feature selector for dimensionality reduction (saved after training, if feature selection is enabled)

### Visualization Files

**`model_comparison.png`** — Visualization showing model performance comparison:
- F1-Score comparison across all models
- Precision comparison
- Accuracy comparison
- Combined metrics visualization
- Highlights ensemble models with red borders

## 3. Instructions on How to Run and Test Your Code

### Prerequisites

Before running the code, ensure you have the following installed:

- **Python 3.7 or higher**
- **pip** (Python package installer)

### How to Set Up the Environment

1. **Install Required Python Packages**

   Install all required dependencies using pip:
   ```bash
   pip install pandas numpy scikit-learn sentence-transformers xgboost joblib matplotlib nltk
   ```

2. **Download NLTK Data (if not automatically downloaded)**

   The code will attempt to automatically download NLTK data, but if needed, you can manually download it:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

   Or run this in a Python shell:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

3. **Verify Required Files**

   Ensure the following files are present in the project directory:
   - `data.csv` (training dataset)
   - `test.csv` (test dataset)

### How to Run Your Labeler

#### Step 1: Train the Model

Train the fraud detection model using the training dataset:

```bash
python policy_proposal_labeler.py
```

Or with custom parameters:
```bash
python policy_proposal_labeler.py "data.csv" True True
```

**Command Arguments:**
1. CSV file path (default: `"data.csv"`)
2. Use ensemble (default: `True`) - Set to `True` to use ensemble models, `False` for single Random Forest
3. Use feature selection (default: `True`) - Set to `True` to enable feature selection, `False` to use all features

**Training Process:**
The script will:
1. Load and preprocess the dataset
2. Extract semantic embeddings and linguistic features
3. Train n-gram model (TF-IDF + XGBoost)
4. Train individual models (Logistic Regression, Random Forest, XGBoost)
5. Create and evaluate base ensemble
6. Create and evaluate final ensemble (with n-gram)
7. Generate performance comparison visualization
8. Save all trained models

**Output files created after training:**
- `fraud_classifier.joblib` - Main ensemble classifier
- `fraud_embedder.joblib` - Sentence transformer model
- `fraud_scaler.joblib` - Feature scaler
- `fraud_ngram_model.joblib` - N-gram model
- `fraud_ngram_vectorizer.joblib` - TF-IDF vectorizer
- `fraud_feature_selector.joblib` - Feature selector (if enabled)
- `model_comparison.png` - Performance visualization

#### Step 2: Make Predictions

After training, use `predictions.py` to make predictions on test cases:

```bash
python predictions.py
```

This script will run predictions on predefined test cases (clear scams, clear non-scams, and ambiguous cases) and display the results with confidence scores.

**Alternative: Using the Labeler Programmatically**

You can also import and use the prediction functions directly in Python:

```python
from policy_proposal_labeler import predict_post, predict_batch

# Single prediction
result = predict_post("Send me your wallet address to claim your prize!")
print(f"Prediction: {result}")

# With probability
pred, prob = predict_post("Send me your wallet address", return_probability=True)
print(f"Prediction: {pred}, Probability: {prob:.4f}")

# Batch prediction
texts = ["Post 1 text...", "Post 2 text...", "Post 3 text..."]
predictions = predict_batch(texts)
probabilities = predict_batch(texts, return_probabilities=True)
```

### How to Run Tests

The evaluation is performed automatically during training. The training script evaluates the model on the test dataset (`test.csv`) and outputs:

1. **Classification Report** - Precision, recall, F1-score for each class
2. **Confusion Matrix** - Detailed breakdown of predictions
3. **ROC-AUC Score** - Area under the ROC curve
4. **Model Comparison Visualization** - Saved as `model_comparison.png`

**To reproduce the evaluation:**

1. Ensure `test.csv` exists in the project directory
2. Run the training script:
   ```bash
   python policy_proposal_labeler.py
   ```
3. The script will automatically:
   - Load the test dataset
   - Evaluate all models (individual and ensemble)
   - Print evaluation metrics to the console
   - Generate and save the comparison visualization

**Expected Output:**
- Console output showing F1-scores, precision, accuracy for each model
- Classification report with per-class metrics
- Confusion matrix
- ROC-AUC score
- Model comparison visualization saved to `model_comparison.png`
