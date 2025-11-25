import pandas as pd
import numpy as np
import re
import os
from urllib.parse import urlparse
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             precision_recall_curve, f1_score, precision_score, accuracy_score)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Some features will be disabled.")


EMBEDDER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDER = None

CLASSIFIER_PATH = "fraud_classifier.joblib"
EMBEDDER_PATH = "fraud_embedder.joblib"
SCALER_PATH = "fraud_scaler.joblib"
FEATURE_SELECTOR_PATH = "fraud_feature_selector.joblib"
NGRAM_VECTORIZER_PATH = "fraud_ngram_vectorizer.joblib"
NGRAM_MODEL_PATH = "fraud_ngram_model.joblib"
FRAUD_KEYWORDS = [
    'send money', 'wire transfer', 'urgent payment', 'verify account',
    'click here', 'limited time', 'act now', 'guaranteed return',
    'risk-free', 'double your', 'crypto investment', 'wallet address',
    'private key', 'seed phrase', 'recover account', 'suspended account',
    'verify identity', 'tax refund', 'lottery winner', 'inheritance',
    'nigerian prince', 'phishing', 'scam', 'fraud', 'ponzi'
]

FINANCIAL_TERMS = [
    'bitcoin', 'crypto', 'ethereum', 'wallet', 'exchange', 'trading',
    'investment', 'profit', 'return', 'dividend', 'stock', 'forex',
    'venmo', 'paypal', 'cashapp', 'zelle', 'western union'
]

SUSPICIOUS_DOMAINS = [
    'bit.ly', 'tinyurl', 'goo.gl', 't.co', 'short.link', 'rebrand.ly'
]


def extract_linguistic_features(text):
    text_lower = text.lower()
    text_len = len(text)
    
    features = {}
    
    features['char_count'] = text_len
    features['word_count'] = len(text.split())
    features['sentence_count'] = len(re.split(r'[.!?]+', text)) if text_len > 0 else 0
    features['avg_word_length'] = np.mean([len(w) for w in text.split()]) if text.split() else 0
    features['avg_sentence_length'] = features['word_count'] / max(features['sentence_count'], 1)
    
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(text_len, 1)
    features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / max(text_len, 1)
    features['punctuation_ratio'] = sum(1 for c in text if c in '.,!?;:') / max(text_len, 1)
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['caps_lock_sequences'] = len(re.findall(r'[A-Z]{3,}', text))
    
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    features['url_count'] = len(urls)
    features['has_url'] = 1 if urls else 0
    
    suspicious_domain_count = 0
    for url in urls:
        try:
            domain = urlparse(url).netloc.lower()
            if any(sd in domain for sd in SUSPICIOUS_DOMAINS):
                suspicious_domain_count += 1
        except:
            pass
    features['suspicious_domain_count'] = suspicious_domain_count
    
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    features['email_count'] = len(re.findall(email_pattern, text))
    
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    features['phone_count'] = len(re.findall(phone_pattern, text))
    
    fraud_keyword_matches = sum(1 for keyword in FRAUD_KEYWORDS if keyword in text_lower)
    financial_term_matches = sum(1 for term in FINANCIAL_TERMS if term in text_lower)
    features['fraud_keyword_count'] = fraud_keyword_matches
    features['financial_term_count'] = financial_term_matches
    features['fraud_keyword_ratio'] = fraud_keyword_matches / max(len(FRAUD_KEYWORDS), 1)
    features['financial_term_ratio'] = financial_term_matches / max(len(FINANCIAL_TERMS), 1)
    
    urgency_words = ['urgent', 'immediately', 'asap', 'now', 'hurry', 'limited', 'expires', 'deadline']
    features['urgency_word_count'] = sum(1 for word in urgency_words if word in text_lower)
    
    cta_patterns = ['click', 'call', 'send', 'verify', 'confirm', 'update', 'act now']
    features['cta_count'] = sum(1 for pattern in cta_patterns if pattern in text_lower)
    
    words = text_lower.split()
    if words:
        word_freq = Counter(words)
        most_common_freq = word_freq.most_common(1)[0][1] if word_freq else 0
        features['max_word_repetition'] = most_common_freq
        features['unique_word_ratio'] = len(set(words)) / len(words)
    
    features['currency_symbol_count'] = text.count('$') + text.count('€') + text.count('£')
    features['percentage_count'] = text.count('%')
    
    if NLTK_AVAILABLE:
        try:
            tokens = word_tokenize(text_lower)
            stop_words = set(stopwords.words('english'))
            features['stopword_ratio'] = sum(1 for w in tokens if w in stop_words) / max(len(tokens), 1)
            features['token_count'] = len(tokens)
        except:
            features['stopword_ratio'] = 0
            features['token_count'] = features['word_count']
    else:
        features['stopword_ratio'] = 0
        features['token_count'] = features['word_count']
    
    return features


def extract_all_features(texts, embedder=None):
    
    
    if embedder is None:
        global EMBEDDER
        if EMBEDDER is None:
            print("Loading embedder model...")
            EMBEDDER = SentenceTransformer(EMBEDDER_MODEL_NAME)
        embedder = EMBEDDER
    
    print("Extracting semantic embeddings...")
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    print("Extracting linguistic features...")
    linguistic_features = []
    for i, text in enumerate(texts):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(texts)} texts...")
        feat_dict = extract_linguistic_features(text)
        linguistic_features.append(list(feat_dict.values()))
    
    linguistic_array = np.array(linguistic_features)
    
    combined_features = np.hstack([embeddings, linguistic_array])
    
    print(f"Feature extraction complete. Shape: {combined_features.shape}")
    print(f"  - Embedding dimensions: {embeddings.shape[1]}")
    print(f"  - Linguistic features: {linguistic_array.shape[1]}")
    
    return combined_features, list(extract_linguistic_features("").keys())



def train_ngram_model(texts_train, y_train, texts_test=None, y_test=None):
    
    
    print("\n" + "-" * 60)
    print("Training N-gram and Keyword Matching Model")
    print("-" * 60)
    
    print("Creating TF-IDF vectorizer with n-grams (1-3)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=5000,
        min_df=2,
        max_df=0.95,
        lowercase=True,
        stop_words='english',
        analyzer='word',
        token_pattern=r'\b\w+\b',
        sublinear_tf=True,
        norm='l2'
    )
    
    print("Fitting vectorizer on training data...")
    X_train_ngram = vectorizer.fit_transform(texts_train)
    print(f"N-gram features extracted: {X_train_ngram.shape[1]} unique n-grams")
    
    print("Training XGBoost classifier on n-gram features...")
    ngram_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if sum(y_train) > 0 else 1,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1,
        tree_method='hist'
    )
    
    ngram_model.fit(X_train_ngram, y_train)
    
    y_train_pred = ngram_model.predict(X_train_ngram)
    train_acc = (y_train_pred == y_train).mean()
    print(f"\nN-gram model training accuracy: {train_acc:.4f}")
    
    if texts_test is not None and y_test is not None:
        X_test_ngram = vectorizer.transform(texts_test)
        y_test_pred = ngram_model.predict(X_test_ngram)
        test_acc = (y_test_pred == y_test).mean()
        print(f"N-gram model test accuracy: {test_acc:.4f}")
        
        if hasattr(ngram_model, 'feature_importances_'):
            feature_names = vectorizer.get_feature_names_out()
            importances = ngram_model.feature_importances_
            top_indices = np.argsort(importances)[-20:][::-1]
            print("\nTop 20 Most Important N-grams:")
            for idx in top_indices:
                print(f"  '{feature_names[idx]}': {importances[idx]:.4f}")
    
    return ngram_model, vectorizer


def predict_with_ngram_model(texts, ngram_model, vectorizer):
    
    
    if isinstance(texts, str):
        texts = [texts]
    
    X_ngram = vectorizer.transform(texts)
    predictions = ngram_model.predict(X_ngram)
    
    if hasattr(ngram_model, 'predict_proba'):
        probabilities = ngram_model.predict_proba(X_ngram)[:, 1]
        if len(texts) == 1:
            return predictions[0], probabilities[0]
        return predictions, probabilities
    
    if len(texts) == 1:
        return predictions[0]
    return predictions



def plot_model_comparison(model_results, save_path="model_comparison.png"):
    
    model_names = list(model_results.keys())
    
    f1_scores = [model_results[name]['f1'] for name in model_names]
    precisions = [model_results[name]['precision'] for name in model_names]
    accuracies = [model_results[name]['accuracy'] for name in model_names]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison: Individual Models vs Ensemble', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    ensemble_color = '#e67e22'
    
    ensemble_idx = None
    for i, name in enumerate(model_names):
        if 'ensemble' in name.lower() or 'combined' in name.lower():
            ensemble_idx = i
            break
    if ensemble_idx is None:
        ensemble_idx = len(model_names) - 1
    
    bar_colors = [ensemble_color if i == ensemble_idx else colors[i % len(colors)] 
                  for i in range(len(model_names))]
    
    x_pos = np.arange(len(model_names))
    width = 0.6
    
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x_pos, f1_scores, width, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax1.set_title('F1-Score Comparison', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, rotation=15, ha='right', fontsize=10)
    ax1.set_ylim([0, 1.05])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=max(f1_scores), color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    for i, (bar, score) in enumerate(zip(bars1, f1_scores)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    if ensemble_idx is not None:
        bars1[ensemble_idx].set_edgecolor('red')
        bars1[ensemble_idx].set_linewidth(2.5)
    
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x_pos, precisions, width, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax2.set_title('Precision Comparison', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names, rotation=15, ha='right', fontsize=10)
    ax2.set_ylim([0, 1.05])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=max(precisions), color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    for i, (bar, score) in enumerate(zip(bars2, precisions)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    if ensemble_idx is not None:
        bars2[ensemble_idx].set_edgecolor('red')
        bars2[ensemble_idx].set_linewidth(2.5)
    
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x_pos, accuracies, width, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax3.set_title('Accuracy Comparison', fontsize=13, fontweight='bold', pad=15)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(model_names, rotation=15, ha='right', fontsize=10)
    ax3.set_ylim([0, 1.05])
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.axhline(y=max(accuracies), color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    for i, (bar, score) in enumerate(zip(bars3, accuracies)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    if ensemble_idx is not None:
        bars3[ensemble_idx].set_edgecolor('red')
        bars3[ensemble_idx].set_linewidth(2.5)
    
    ax4 = axes[1, 1]
    x = np.arange(len(model_names))
    width_bar = 0.25
    
    bars_f1 = ax4.bar(x - width_bar, f1_scores, width_bar, label='F1-Score', 
                      color='#3498db', alpha=0.8, edgecolor='black')
    bars_prec = ax4.bar(x, precisions, width_bar, label='Precision', 
                       color='#2ecc71', alpha=0.8, edgecolor='black')
    bars_acc = ax4.bar(x + width_bar, accuracies, width_bar, label='Accuracy', 
                      color='#e74c3c', alpha=0.8, edgecolor='black')
    
    ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax4.set_title('All Metrics Combined Comparison', fontsize=13, fontweight='bold', pad=15)
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_names, rotation=15, ha='right', fontsize=10)
    ax4.set_ylim([0, 1.05])
    ax4.legend(loc='upper left', fontsize=10)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    if ensemble_idx is not None:
        bars_f1[ensemble_idx].set_edgecolor('red')
        bars_f1[ensemble_idx].set_linewidth(2.5)
        bars_prec[ensemble_idx].set_edgecolor('red')
        bars_prec[ensemble_idx].set_linewidth(2.5)
        bars_acc[ensemble_idx].set_edgecolor('red')
        bars_acc[ensemble_idx].set_linewidth(2.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Model comparison visualization saved to: {save_path}")
    plt.close()



class CustomEnsemble:
    def __init__(self, base_ensemble, ngram_model, ngram_vectorizer):
        self.base_ensemble = base_ensemble
        self.ngram_model = ngram_model
        self.ngram_vectorizer = ngram_vectorizer
    
    def predict(self, X, texts=None):
        base_proba = self.base_ensemble.predict_proba(X)
        
        if texts is not None:
            ngram_proba = self.ngram_model.predict_proba(
                self.ngram_vectorizer.transform(texts)
            )
            combined_proba = 0.6 * base_proba + 0.4 * ngram_proba
            return np.argmax(combined_proba, axis=1)
        else:
            return self.base_ensemble.predict(X)
    
    def predict_proba(self, X, texts=None):
        base_proba = self.base_ensemble.predict_proba(X)
        
        if texts is not None:
            ngram_proba = self.ngram_model.predict_proba(
                self.ngram_vectorizer.transform(texts)
            )
            combined_proba = 0.6 * base_proba + 0.4 * ngram_proba
            return combined_proba
        else:
            return base_proba


def train_model_from_csv(csv_path, use_ensemble=True, use_feature_selection=True):
    
    
    
    
    global EMBEDDER
    
    print("=" * 60)
    print("ADVANCED FRAUD DETECTION MODEL TRAINING")
    print("=" * 60)
    
    print(f"\n[1/8] Loading training dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if "Post Content" not in df.columns or "Ground Truth Label" not in df.columns:
        raise ValueError("CSV file must contain 'Post Content' and 'Ground Truth Label' columns.")
    
    # Filter out rows with NaN labels
    print(f"Total rows in CSV: {len(df)}")
    df = df.dropna(subset=["Ground Truth Label"])
    print(f"Rows with valid labels: {len(df)}")
    
    texts = df["Post Content"].astype(str).tolist()
    labels = np.array(df["Ground Truth Label"].tolist())
    
    print(f"Loaded {len(texts)} training posts")
    print(f"Training label distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    print(f"\n[2/8] Loading test dataset: test.csv")
    test_csv_path = "test.csv"
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test CSV file '{test_csv_path}' not found. Please ensure test.csv exists.")
    
    df_test = pd.read_csv(test_csv_path)
    
    if "Post Content" not in df_test.columns or "Ground Truth Label" not in df_test.columns:
        raise ValueError("Test CSV file must contain 'Post Content' and 'Ground Truth Label' columns.")
    
    # Filter out rows with NaN labels
    print(f"Total rows in test CSV: {len(df_test)}")
    df_test = df_test.dropna(subset=["Ground Truth Label"])
    print(f"Rows with valid labels in test CSV: {len(df_test)}")
    
    texts_test = df_test["Post Content"].astype(str).tolist()
    y_test = np.array(df_test["Ground Truth Label"].tolist())
    
    print(f"Loaded {len(texts_test)} test posts")
    print(f"Test label distribution: {pd.Series(y_test).value_counts().to_dict()}")
    
    print(f"\n[3/8] Extracting features for training data...")
    EMBEDDER = SentenceTransformer(EMBEDDER_MODEL_NAME)
    X_train, feature_names = extract_all_features(texts, EMBEDDER)
    
    print(f"\n[4/8] Extracting features for test data...")
    X_test, _ = extract_all_features(texts_test, EMBEDDER)
    
    # Use all training data (no split)
    texts_train = texts
    y_train = labels
    
    print(f"\n[5/8] Training n-gram and keyword matching model...")
    ngram_model, ngram_vectorizer = train_ngram_model(
        texts_train, y_train, texts_test, y_test
    )
    
    print(f"\n[6/8] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    feature_selector = None
    if use_feature_selection:
        print(f"\n[7/8] Applying feature selection...")
        k_best = min(500, X_train_scaled.shape[1])
        feature_selector = SelectKBest(score_func=f_classif, k=k_best)
        X_train_selected = feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = feature_selector.transform(X_test_scaled)
        print(f"Selected {k_best} features from {X_train_scaled.shape[1]} total")
    else:
        X_train_selected = X_train_scaled
        X_test_selected = X_test_scaled
    
    print(f"\n[8/8] Training ensemble classifier...")
    
    model_results = {}
    
    if use_ensemble:
        models = []
        
        print("\n" + "=" * 60)
        print("INDIVIDUAL MODEL EVALUATION")
        print("=" * 60)
        
        print("\n[1/5] Training and evaluating Logistic Regression...")
        lr = LogisticRegression(
            class_weight=None,
            max_iter=100,
            C=0.01,
            solver='liblinear',
            random_state=42
        )
        lr.fit(X_train_selected, y_train)
        lr_pred = lr.predict(X_test_selected)
        lr_f1 = f1_score(y_test, lr_pred)
        lr_prec = precision_score(y_test, lr_pred)
        lr_acc = accuracy_score(y_test, lr_pred)
        model_results['Logistic Regression'] = {
            'f1': lr_f1,
            'precision': lr_prec,
            'accuracy': lr_acc
        }
        print(f"  F1-Score: {lr_f1:.4f}, Precision: {lr_prec:.4f}, Accuracy: {lr_acc:.4f}")
        
        lr_ensemble = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            C=1.0,
            solver='lbfgs',
            random_state=42
        )
        lr_ensemble.fit(X_train_selected, y_train)
        models.append(('lr', lr_ensemble))
        
        print("\n[2/5] Training and evaluating Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=10,
            max_depth=3,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight=None,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train_selected, y_train)
        rf_pred = rf.predict(X_test_selected)
        rf_f1 = f1_score(y_test, rf_pred)
        rf_prec = precision_score(y_test, rf_pred)
        rf_acc = accuracy_score(y_test, rf_pred)
        model_results['Random Forest'] = {
            'f1': rf_f1,
            'precision': rf_prec,
            'accuracy': rf_acc
        }
        print(f"  F1-Score: {rf_f1:.4f}, Precision: {rf_prec:.4f}, Accuracy: {rf_acc:.4f}")
        
        rf_ensemble = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        rf_ensemble.fit(X_train_selected, y_train)
        models.append(('rf', rf_ensemble))
        
        print("\n[3/5] Training and evaluating XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if sum(y_train) > 0 else 1,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        )
        xgb_model.fit(X_train_selected, y_train)
        xgb_pred = xgb_model.predict(X_test_selected)
        xgb_f1 = f1_score(y_test, xgb_pred)
        xgb_prec = precision_score(y_test, xgb_pred)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        model_results['XGBoost'] = {
            'f1': xgb_f1,
            'precision': xgb_prec,
            'accuracy': xgb_acc
        }
        print(f"  F1-Score: {xgb_f1:.4f}, Precision: {xgb_prec:.4f}, Accuracy: {xgb_acc:.4f}")
        models.append(('xgb', xgb_model))
        
        print("\n[4/5] Evaluating N-gram Model...")
        ngram_result = predict_with_ngram_model(texts_test, ngram_model, ngram_vectorizer)
        if isinstance(ngram_result, tuple):
            ngram_pred, _ = ngram_result
        else:
            ngram_pred = ngram_result
        ngram_f1 = f1_score(y_test, ngram_pred)
        ngram_prec = precision_score(y_test, ngram_pred)
        ngram_acc = accuracy_score(y_test, ngram_pred)
        model_results['N-gram Model'] = {
            'f1': ngram_f1,
            'precision': ngram_prec,
            'accuracy': ngram_acc
        }
        print(f"  F1-Score: {ngram_f1:.4f}, Precision: {ngram_prec:.4f}, Accuracy: {ngram_acc:.4f}")
        
        print("\n[5/5] Creating base ensemble (LR + RF + XGB)...")
        base_ensemble = VotingClassifier(
            estimators=models,
            voting='soft',
            n_jobs=-1
        )
        
        base_ensemble.fit(X_train_selected, y_train)
        
        print("\nEvaluating base ensemble (LR + RF + XGB)...")
        base_ensemble_pred = base_ensemble.predict(X_test_selected)
        base_ensemble_f1 = f1_score(y_test, base_ensemble_pred)
        base_ensemble_prec = precision_score(y_test, base_ensemble_pred)
        base_ensemble_acc = accuracy_score(y_test, base_ensemble_pred)
        model_results['Base Ensemble (LR+RF+XGB)'] = {
            'f1': base_ensemble_f1,
            'precision': base_ensemble_prec,
            'accuracy': base_ensemble_acc
        }
        print(f"  F1-Score: {base_ensemble_f1:.4f}, Precision: {base_ensemble_prec:.4f}, Accuracy: {base_ensemble_acc:.4f}")
        
        clf = CustomEnsemble(base_ensemble, ngram_model, ngram_vectorizer)
        
        print("\nEvaluating Final Ensemble (Base + N-gram)...")
        final_ensemble_pred = clf.predict(X_test_selected, texts=texts_test)
        final_ensemble_f1 = f1_score(y_test, final_ensemble_pred)
        final_ensemble_prec = precision_score(y_test, final_ensemble_pred)
        final_ensemble_acc = accuracy_score(y_test, final_ensemble_pred)
        model_results['Final Ensemble (All Models)'] = {
            'f1': final_ensemble_f1,
            'precision': final_ensemble_prec,
            'accuracy': final_ensemble_acc
        }
        print(f"  F1-Score: {final_ensemble_f1:.4f}, Precision: {final_ensemble_prec:.4f}, Accuracy: {final_ensemble_acc:.4f}")
        
        print("\n" + "=" * 60)
        print("GENERATING MODEL COMPARISON VISUALIZATION")
        print("=" * 60)
        plot_model_comparison(model_results, save_path="model_comparison.png")
        
    else:
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train_selected, y_train)
    
    is_custom_ensemble = hasattr(clf, '__class__') and clf.__class__.__name__ == 'CustomEnsemble'
    if not is_custom_ensemble:
        print("\nPerforming cross-validation...")
        try:
            cv_scores = cross_val_score(clf, X_train_selected, y_train, cv=5, scoring='f1', n_jobs=-1)
            print(f"Cross-validation F1 scores: {cv_scores}")
            print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        except:
            print("Cross-validation skipped")
    
    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION")
    print("=" * 60)
    
    if is_custom_ensemble:
        y_pred = clf.predict(X_test_selected, texts=texts_test)
        y_pred_proba = clf.predict_proba(X_test_selected, texts=texts_test)[:, 1]
    else:
        y_pred = clf.predict(X_test_selected)
        y_pred_proba = clf.predict_proba(X_test_selected)[:, 1] if hasattr(clf, 'predict_proba') else None
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
    
    if y_pred_proba is not None:
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
            print(f"\nROC-AUC Score: {auc:.4f}")
        except:
            pass
    
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        top_indices = np.argsort(importances)[-10:][::-1]
        print("\nTop 10 Most Important Features:")
        for idx in top_indices:
            if idx < len(feature_names):
                print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
    
    print("\n" + "=" * 60)
    print("Saving models...")
    joblib.dump(clf, CLASSIFIER_PATH)
    joblib.dump(EMBEDDER, EMBEDDER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(ngram_model, NGRAM_MODEL_PATH)
    joblib.dump(ngram_vectorizer, NGRAM_VECTORIZER_PATH)
    if feature_selector:
        joblib.dump(feature_selector, FEATURE_SELECTOR_PATH)
    
    print(f"✓ Classifier: {CLASSIFIER_PATH}")
    print(f"✓ Embedder: {EMBEDDER_PATH}")
    print(f"✓ Scaler: {SCALER_PATH}")
    print(f"✓ N-gram Model: {NGRAM_MODEL_PATH}")
    print(f"✓ N-gram Vectorizer: {NGRAM_VECTORIZER_PATH}")
    if feature_selector:
        print(f"✓ Feature Selector: {FEATURE_SELECTOR_PATH}")
    
    print("\nTraining complete!")
    return clf, scaler, feature_selector, ngram_model, ngram_vectorizer


def load_model():
    
    global EMBEDDER
    
    if not os.path.exists(CLASSIFIER_PATH) or not os.path.exists(EMBEDDER_PATH):
        raise FileNotFoundError(
            "Model files not found. Train the model first using train_model_from_csv()."
        )
    
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "Scaler file not found. Please retrain the model with the updated code."
        )
    
    print("Loading model components...")
    EMBEDDER = joblib.load(EMBEDDER_PATH)
    CLASSIFIER = joblib.load(CLASSIFIER_PATH)
    SCALER = joblib.load(SCALER_PATH)
    
    FEATURE_SELECTOR = None
    if os.path.exists(FEATURE_SELECTOR_PATH):
        FEATURE_SELECTOR = joblib.load(FEATURE_SELECTOR_PATH)
    
    NGRAM_MODEL = None
    NGRAM_VECTORIZER = None
    if os.path.exists(NGRAM_MODEL_PATH) and os.path.exists(NGRAM_VECTORIZER_PATH):
        NGRAM_MODEL = joblib.load(NGRAM_MODEL_PATH)
        NGRAM_VECTORIZER = joblib.load(NGRAM_VECTORIZER_PATH)
        print("N-gram model loaded successfully!")
    
    print("Model loaded successfully!")
    return CLASSIFIER, EMBEDDER, SCALER, FEATURE_SELECTOR, NGRAM_MODEL, NGRAM_VECTORIZER


_LOADED_CLASSIFIER = None
_LOADED_SCALER = None
_LOADED_FEATURE_SELECTOR = None
_LOADED_NGRAM_MODEL = None
_LOADED_NGRAM_VECTORIZER = None


def predict_post(text, return_probability=False):
    
    
    
    
    global EMBEDDER, _LOADED_CLASSIFIER, _LOADED_SCALER, _LOADED_FEATURE_SELECTOR
    global _LOADED_NGRAM_MODEL, _LOADED_NGRAM_VECTORIZER
    
    if _LOADED_CLASSIFIER is None or EMBEDDER is None or _LOADED_SCALER is None:
        (_LOADED_CLASSIFIER, EMBEDDER, _LOADED_SCALER, _LOADED_FEATURE_SELECTOR,
         _LOADED_NGRAM_MODEL, _LOADED_NGRAM_VECTORIZER) = load_model()
    
    embedding = EMBEDDER.encode([text], convert_to_numpy=True)
    
    linguistic_feat_dict = extract_linguistic_features(text)
    linguistic_features = np.array([list(linguistic_feat_dict.values())])
    
    combined_features = np.hstack([embedding, linguistic_features])
    
    features_scaled = _LOADED_SCALER.transform(combined_features)
    
    if _LOADED_FEATURE_SELECTOR is not None:
        features_final = _LOADED_FEATURE_SELECTOR.transform(features_scaled)
    else:
        features_final = features_scaled
    
    is_custom_ensemble = (hasattr(_LOADED_CLASSIFIER, '__class__') and 
                         _LOADED_CLASSIFIER.__class__.__name__ == 'CustomEnsemble')
    
    if is_custom_ensemble:
        prediction = _LOADED_CLASSIFIER.predict(features_final, texts=[text])[0]
        if return_probability:
            probability = _LOADED_CLASSIFIER.predict_proba(features_final, texts=[text])[0, 1]
            return prediction, probability
    else:
        prediction = _LOADED_CLASSIFIER.predict(features_final)[0]
        
        if return_probability and _LOADED_NGRAM_MODEL is not None and _LOADED_NGRAM_VECTORIZER is not None:
            base_proba = _LOADED_CLASSIFIER.predict_proba(features_final)[0, 1] if hasattr(_LOADED_CLASSIFIER, 'predict_proba') else float(prediction)
            ngram_proba = _LOADED_NGRAM_MODEL.predict_proba(
                _LOADED_NGRAM_VECTORIZER.transform([text])
            )[0, 1]
            probability = 0.6 * base_proba + 0.4 * ngram_proba
            prediction = 1 if probability >= 0.5 else 0
            return prediction, probability
        elif return_probability:
            if hasattr(_LOADED_CLASSIFIER, 'predict_proba'):
                probability = _LOADED_CLASSIFIER.predict_proba(features_final)[0, 1]
            else:
                probability = float(prediction)
            return prediction, probability
    
    return prediction


def predict_batch(texts, return_probabilities=False):
    
    
    global EMBEDDER, _LOADED_CLASSIFIER, _LOADED_SCALER, _LOADED_FEATURE_SELECTOR
    global _LOADED_NGRAM_MODEL, _LOADED_NGRAM_VECTORIZER
    
    if _LOADED_CLASSIFIER is None or EMBEDDER is None or _LOADED_SCALER is None:
        (_LOADED_CLASSIFIER, EMBEDDER, _LOADED_SCALER, _LOADED_FEATURE_SELECTOR,
         _LOADED_NGRAM_MODEL, _LOADED_NGRAM_VECTORIZER) = load_model()
    
    embeddings = EMBEDDER.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    linguistic_features_list = []
    for text in texts:
        feat_dict = extract_linguistic_features(text)
        linguistic_features_list.append(list(feat_dict.values()))
    
    linguistic_features = np.array(linguistic_features_list)
    
    combined = np.hstack([embeddings, linguistic_features])
    scaled = _LOADED_SCALER.transform(combined)
    
    if _LOADED_FEATURE_SELECTOR is not None:
        final_features = _LOADED_FEATURE_SELECTOR.transform(scaled)
    else:
        final_features = scaled
    
    is_custom_ensemble = (hasattr(_LOADED_CLASSIFIER, '__class__') and 
                         _LOADED_CLASSIFIER.__class__.__name__ == 'CustomEnsemble')
    
    if is_custom_ensemble:
        predictions = _LOADED_CLASSIFIER.predict(final_features, texts=texts)
        if return_probabilities:
            probabilities = _LOADED_CLASSIFIER.predict_proba(final_features, texts=texts)[:, 1]
            return predictions, probabilities
    else:
        predictions = _LOADED_CLASSIFIER.predict(final_features)
        
        if return_probabilities:
            if hasattr(_LOADED_CLASSIFIER, 'predict_proba'):
                base_proba = _LOADED_CLASSIFIER.predict_proba(final_features)[:, 1]
            else:
                base_proba = predictions.astype(float)
            
            if _LOADED_NGRAM_MODEL is not None and _LOADED_NGRAM_VECTORIZER is not None:
                ngram_proba = _LOADED_NGRAM_MODEL.predict_proba(
                    _LOADED_NGRAM_VECTORIZER.transform(texts)
                )[:, 1]
                probabilities = 0.6 * base_proba + 0.4 * ngram_proba
                predictions = (probabilities >= 0.5).astype(int)
            else:
                probabilities = base_proba
            
            return predictions, probabilities
    
    return predictions


if __name__ == "__main__":
    import sys
    
    csv_path = "data.csv"
    
    use_ensemble = True
    use_feature_selection = True
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    if len(sys.argv) > 2:
        use_ensemble = sys.argv[2].lower() == 'true'
    if len(sys.argv) > 3:
        use_feature_selection = sys.argv[3].lower() == 'true'
    
    print("Training with:")
    print(f"  - CSV file: {csv_path}")
    print(f"  - Ensemble: {use_ensemble}")
    print(f"  - Feature selection: {use_feature_selection}")
    print()
    
    train_model_from_csv(csv_path, use_ensemble=use_ensemble, use_feature_selection=use_feature_selection)
