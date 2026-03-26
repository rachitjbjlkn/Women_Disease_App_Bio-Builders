"""Training routines for tabular and image models."""

from typing import Tuple
import numpy as np
import pandas as pd


def train_classical_model(X, y, model):
    """Train a scikit-learn estimator and return it."""
    model.fit(X, y)
    return model


def train_and_evaluate_tabular(df, test_size=0.2, random_state=42):
    """Train multiple best algorithms and return results.
    df must include a 'target' column.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
    from sklearn.svm import SVC
    import numpy as _np

    # Try to import xgboost, but don't fail if it's not available
    try:
        from xgboost import XGBClassifier
        has_xgboost = True
    except ImportError:
        has_xgboost = False

    df_proc = df.copy()
    df_proc = df_proc.dropna(subset=["target"])
    y = df_proc["target"]
    X = df_proc.drop("target", axis=1)

    for col in X.select_dtypes(include=[_np.number]).columns:
        X[col] = X[col].fillna(X[col].median())
    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models_dict = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=random_state, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=random_state),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=15, random_state=random_state, n_jobs=-1),
        "SVM": SVC(kernel='rbf', probability=True, random_state=random_state)
    }
    
    # Add XGBoost if available
    if has_xgboost:
        models_dict["XGBoost"] = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=random_state, verbosity=0)
    
    # --- ENSEMBLE CLASSIFIER (Upgraded Model to Prevent Wrong Predictions) ---
    estimators = [
        ('rf', models_dict["Random Forest"]),
        ('gb', models_dict["Gradient Boosting"]),
        ('et', models_dict["Extra Trees"])
    ]
    if has_xgboost:
        estimators.append(('xgb', models_dict["XGBoost"]))
        
    models_dict["Ensemble (Robust Voting)"] = VotingClassifier(estimators=estimators, voting='soft')
    
    trained = {}
    metrics_dict = {}
    for name, model in models_dict.items():
        model.fit(X_train, y_train)
        trained[name] = model
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = None
        metrics_dict[name] = (y_test, y_pred, y_prob)
    
    return trained, metrics_dict, scaler, X_train, X_test, y_train, y_test


def train_cnn_model(X, y, input_shape: Tuple[int, int, int]):
    """Build and train a simple CNN using Keras. You can swap for transfer learning."""
    from tensorflow.keras import models, layers, optimizers

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
    return model


def train_image_classifier(X, y, input_shape=(64, 64, 3), num_epochs=5, batch_size=32):
    """Train a CNN classifier on image arrays using a simple architecture or transfer learning."""
    from tensorflow.keras import models, layers, optimizers
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import numpy as _np

    # encode labels
    classes, y_indices = _np.unique(y, return_inverse=True)
    y_cat = to_categorical(y_indices, num_classes=len(classes))

    # data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )

    # use transfer learning with EfficientNetB0
    try:
        from tensorflow.keras.applications import EfficientNetB0
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False  # freeze base
        model = models.Sequential([
            layers.Input(shape=input_shape),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(classes), activation='softmax')
        ])
    except:
        # fallback to simple CNN
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3,3), activation='relu'),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(classes), activation='softmax')
        ])

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # train with augmentation
    history = model.fit(
        datagen.flow(X, y_cat, batch_size=batch_size),
        epochs=num_epochs,
        validation_split=0.2,
        verbose=0
    )
    return model, classes, history


def evaluate_model(model, X_test, y_test, is_image=False):
    """Compute predictions and standard metrics for a model. Handles both sklearn and keras."""
    import numpy as _np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

    if hasattr(model, 'predict_proba') or hasattr(model, 'predict') and not is_image:
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = None
    else:
        # assume keras model
        y_prob_all = model.predict(X_test)
        if y_prob_all.shape[-1] > 1:
            y_pred = _np.argmax(y_prob_all, axis=1)
            y_prob = None
        else:
            y_pred = (y_prob_all > 0.5).astype(int).flatten()
            y_prob = y_prob_all.flatten()
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    if y_prob is not None and len(_np.unique(y_test)) == 2:
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        except:
            pass
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
    return metrics
