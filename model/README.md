# BuildingDamage-Assessment Model Files

## Model Structure

This directory contains the hybrid CNN + Random Forest model for temporal building damage assessment:

### Files:
- `temporal_cnn_weights.h5` - CNN feature extractor (4.4 MB)
- `temporal_rf.pkl` - Random Forest classifier (61 KB)

## Loading the Models

```python
import pickle
from tensorflow import keras

# Load CNN feature extractor
cnn_model = keras.models.load_model('temporal_cnn_weights.h5')

# Load Random Forest classifier
with open('temporal_rf.pkl', 'rb') as f:
    rf_classifier = pickle.load(f)
```

## Usage Example

```python
import numpy as np
from PIL import Image

def assess_damage(before_image, after_image):
    # Preprocess images
    before = np.array(before_image.resize((224, 224))) / 255.0
    after = np.array(after_image.resize((224, 224))) / 255.0

    # Combine temporal data
    combined = np.stack([before, after])
    combined = np.expand_dims(combined, axis=0)

    # Extract features
    features = cnn_model.predict(combined)

    # Classify damage
    damage_level = rf_classifier.predict(features)[0]

    return ['No Damage', 'Minor', 'Major', 'Destroyed'][damage_level]
```

## Model Specifications

- **CNN Model Size**: 4.4 MB (Keras H5 format)
- **RF Model Size**: 61 KB (Pickle format)
- **Input**: Before/after image pairs (224x224 RGB)
- **Output**: 4-class damage classification
- **Performance**: 73.56% training accuracy, 73.39% validation accuracy