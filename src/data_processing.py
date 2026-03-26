# Data loading and preprocessing utilities for the woman disease project

import pandas as pd
import numpy as np


def load_tabular_data(path: str) -> pd.DataFrame:
    """Read a CSV file containing clinical/text features."""
    df = pd.read_csv(path)
    return df


def generate_synthetic_classification(n_samples=2000, n_features=10, n_informative=5, random_state=42):
    """Utility to create a synthetic classification dataset."""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_informative=n_informative,
                               n_redundant=0,
                               n_clusters_per_class=1,
                               random_state=random_state)
    cols = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    return df


def get_ovarian_cyst_data() -> pd.DataFrame:
    """Load or generate a sample ovarian cyst dataset."""
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "data", "ovarian_cyst.csv")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = generate_synthetic_classification(n_samples=2000, n_features=8, n_informative=4)
        # give some realistic column names
        df.columns = ["age", "menstrual_cycle", "hormone_level", "bmi", "symptom_score", "ultrasound_score", "previous_cysts", "family_history", "target"]
        df.to_csv(path, index=False)
    return df


def get_pcos_data() -> pd.DataFrame:
    """Load or generate a sample PCOS dataset."""
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "data", "pcos.csv")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = generate_synthetic_classification(n_samples=2500, n_features=10, n_informative=6)
        df.columns = ["age", "weight", "insulin_level", "testosterone", "glucose", "bmi", "blood_pressure", "cycle_regular", "acne", "hair_growth", "target"]
        df.to_csv(path, index=False)
    return df


def get_breast_cancer_data() -> pd.DataFrame:
    """Load breast cancer data from sklearn if available, else generate."""
    from sklearn.datasets import load_breast_cancer
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "data", "breast_cancer.csv")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        bc = load_breast_cancer(as_frame=True)
        df = bc.frame
        df.to_csv(path, index=False)
    return df


def preprocess_tabular(df: pd.DataFrame) -> pd.DataFrame:
    """Basic preprocessing: imputation, encoding and scaling.
    Expand this function as requirements grow.
    """
    df = df.copy()
    # fill numeric missing values with median
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)

    # example: one-hot encode categorical columns
    df = pd.get_dummies(df, drop_first=True)
    return df


def load_image(path: str, target_size=(224, 224)) -> np.ndarray:
    """Load single image from path and resize to target_size."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img) / 255.0
    return arr


def load_image_dataset(folder: str, target_size=(224, 224)):
    """Walk a directory where subfolders are class names and load images and labels."""
    import os
    from PIL import Image
    X, y = [], []
    classes = []
    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):
            classes.append(class_name)
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('png','jpg','jpeg','bmp')):
                    img = Image.open(os.path.join(class_path, fname)).convert('RGB')
                    img = img.resize(target_size)
                    X.append(np.array(img)/255.0)
                    y.append(class_name)
    return np.array(X), np.array(y), classes


def get_synthetic_image_data(n_samples=1000, image_size=(64, 64), num_classes=2):
    """Return synthetic image data using CIFAR10 filtered to two classes for demo."""
    from tensorflow.keras.datasets import cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    import numpy as _np
    X = _np.concatenate([X_train, X_test])
    y = _np.concatenate([y_train, y_test]).flatten()
    # pick two classes randomly for binary classification
    classes = _np.unique(y)
    if len(classes) < num_classes:
        return X, y
    chosen = classes[:num_classes]
    mask = _np.isin(y, chosen)
    X = X[mask]
    y = y[mask]
    # resize if needed
    from tensorflow.image import resize
    X = resize(X, image_size).numpy()
    # normalize
    X = X.astype('float32')/255.0
    return X, y


def validate_ultrasound_image(file_obj) -> tuple[bool, str]:
    """Validate if the uploaded file appears to be a genuine ultrasound image.
    Uses heuristics like checking color saturation (ultrasounds are mostly grayscale)."""
    try:
        from PIL import Image
        import numpy as np
        import io
        
        # Read the file bytes
        file_bytes = file_obj.read()
        file_obj.seek(0)  # Reset pointer for subsequent reads
        
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        arr = np.array(img)
        
        # Heuristic 1: Strict Color check - Grayscale images have extremely low color variance.
        # R, G, B standard deviation over all pixels.
        color_std = np.std(arr, axis=2)
        mean_std = np.mean(color_std)
        
        # Stricter threshold: Any average saturation > 5.0 is almost definitely a normal photo, not an ultrasound scan.
        if mean_std > 5.0:
            return False, "❌ Validation Failed: The uploaded image is not a valid medical structure. Ensure you only upload grayscale ultrasound or scans."
        
        # Heuristic 2: Absolute Uniformity or Brightness Check (Blank images, pitch black, or overexposed whites)
        mean_brightness = np.mean(arr)
        if mean_brightness < 10 or mean_brightness > 220:
            return False, "❌ Validation Failed: The image lighting is incompatible with medical ultrasound standards."
            
        # Heuristic 3: Check for Dark Background Dominance
        # Ultrasounds typically have a large amount of black/near-black background
        # Let's check if at least 15% of the image is very dark (< 30 pixel value)
        dark_pixels_ratio = np.sum(arr < 30) / arr.size
        
        if dark_pixels_ratio < 0.15:
            return False, "❌ Validation Failed: Image lacks the characteristic dark background of a medical ultrasound."
            
        # Heuristic 4: Texture/Speckle presence
        overall_std = np.std(arr)
        if overall_std < 15:
            return False, "❌ Validation Failed: Image lacks the structural texture/speckle of an ultrasound scan."
            
        return True, "✅ Ultrasound image structure strictly validated successfully."
    except Exception as e:
        return False, f"❌ Validation Error: Could not process or read the file properly. ({str(e)})"
