# CS 5342 Assignment 3: Automated Moderator (Bluesky Labeler)

## Files Submitted

### 1. `policy_proposal_labeler.py`
The main implementation file containing:
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

### 2. `data.csv`
Training dataset containing:
- **Post Content**: Text content of Bluesky posts
- **Ground Truth Label**: Binary labels (0 = Legitimate, 1 = Fraud)

### 3. `model_comparison.png`
Visualization showing model performance comparison:
- F1-Score comparison across all models
- Precision comparison
- Accuracy comparison
- Combined metrics visualization
- Highlights ensemble models with red borders

## Dependencies

Install required packages:
```bash
pip install pandas numpy scikit-learn sentence-transformers xgboost joblib matplotlib nltk
```

For NLTK data (if not automatically downloaded):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## How to Run

### Step 1: Train the Model
Run the training script:
```bash
python policy_proposal_labeler.py
```

Or with custom parameters:
```bash
python policy_proposal_labeler.py "data.csv" True True
```
Arguments:
1. CSV file path (default: "data.csv")
2. Use ensemble (default: True)
3. Use feature selection (default: True)

### Step 3: Training Process
The script will:
1. Load and preprocess the dataset
2. Extract semantic embeddings and linguistic features
3. Train n-gram model (TF-IDF + XGBoost)
4. Train individual models (Logistic Regression, Random Forest, XGBoost)
5. Create and evaluate base ensemble
6. Create and evaluate final ensemble (with n-gram)
7. Generate performance comparison visualization
8. Save all trained models

**Output files created:**
- `fraud_classifier.joblib` - Main ensemble classifier
- `fraud_embedder.joblib` - Sentence transformer model
- `fraud_scaler.joblib` - Feature scaler
- `fraud_ngram_model.joblib` - N-gram model
- `fraud_ngram_vectorizer.joblib` - TF-IDF vectorizer
- `fraud_feature_selector.joblib` - Feature selector (if enabled)
- `model_comparison.png` - Performance visualization

### Step 4: Make Predictions
After training, use the model to predict on new posts:

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

## Model Architecture

### Feature Extraction
1. **Semantic Embeddings** (384 dims): Uses `sentence-transformers/all-MiniLM-L6-v2`
2. **Linguistic Features** (28 dims):
   - Text statistics (char/word/sentence counts)
   - Character patterns (uppercase ratio, punctuation, etc.)
   - URL/domain analysis
   - Email/phone detection
   - Fraud keyword matching
   - Financial term detection
   - Urgency indicators
   - Call-to-action patterns
   - Repetition detection
   - Special characters

### Models
1. **Logistic Regression**: Linear classifier (suboptimal version for comparison)
2. **Random Forest**: Tree-based ensemble (suboptimal version for comparison)
3. **XGBoost**: Gradient boosting classifier
4. **N-gram Model**: TF-IDF (1-3 grams) + XGBoost
5. **Base Ensemble**: Voting classifier (LR + RF + XGB)
6. **Final Ensemble**: Weighted combination (60% base + 40% n-gram)

### Evaluation Metrics
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: Proportion of predicted frauds that are actually fraud
- **Accuracy**: Overall correctness
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: Detailed breakdown of predictions

## Testing and Evaluation

The model is evaluated on a held-out test set (20% of data):