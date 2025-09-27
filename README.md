# BuildingDamage-Assessment Model

## Temporal Building Damage Evaluation for Conflict Zones

Advanced deep learning model for assessing building damage severity using before/after satellite imagery pairs. Essential for humanitarian response, damage documentation, and legal proceedings.

## Model Overview

- **Architecture**: Hybrid CNN + Random Forest
- **Model Size**: 4.5 MB (highly optimized)
- **Input**: Before/after image pairs
- **Output**: 4-class damage classification
- **Training Data**: 162,787 validated samples from 2,799 disaster events

## Performance Metrics

| Metric | Score |
|--------|-------|
| Training Accuracy | 73.56% |
| Validation Accuracy | 73.39% |
| Training Loss | 0.8677 |
| Validation Loss | 0.8713 |
| Inference Speed | < 100ms per image pair |

## Damage Classification Levels

1. **No Damage** - Building intact, no visible damage
2. **Minor Damage** - Superficial damage, structure intact
3. **Major Damage** - Significant structural damage, partially collapsed
4. **Destroyed** - Complete destruction, rubble only

## Installation

```bash
pip install tensorflow scikit-learn opencv-python pillow numpy
```

## Model Files

```
model/
├── temporal_cnn_weights.h5    # CNN feature extractor (4.4 MB)
├── temporal_rf.pkl            # Random Forest classifier (61 KB)
└── training_metadata.json     # Model configuration (2 KB)
```

## Usage

### Basic Usage

```python
import pickle
from tensorflow import keras
import numpy as np
from PIL import Image

# Load models
cnn_model = keras.models.load_model('temporal_cnn_weights.h5')
with open('temporal_rf.pkl', 'rb') as f:
    rf_classifier = pickle.load(f)

def assess_damage(before_image_path, after_image_path):
    # Load and preprocess images
    before = Image.open(before_image_path).resize((224, 224))
    after = Image.open(after_image_path).resize((224, 224))

    before_array = np.array(before) / 255.0
    after_array = np.array(after) / 255.0

    # Stack images for temporal analysis
    combined = np.stack([before_array, after_array])
    combined = np.expand_dims(combined, axis=0)

    # Extract features using CNN
    features = cnn_model.predict(combined)

    # Classify damage level
    damage_level = rf_classifier.predict(features)[0]

    damage_labels = ['No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']
    return damage_labels[damage_level]

# Example usage
damage = assess_damage('before.jpg', 'after.jpg')
print(f"Damage Assessment: {damage}")
```

### Batch Processing

```python
def batch_assess(image_pairs):
    results = []
    for before_path, after_path in image_pairs:
        damage = assess_damage(before_path, after_path)
        results.append({
            'before': before_path,
            'after': after_path,
            'damage_level': damage
        })
    return results
```

## Applications

### Humanitarian Response
- **Rapid Damage Assessment**: Quick evaluation for emergency response
- **Resource Allocation**: Prioritize aid based on damage severity
- **Recovery Planning**: Track reconstruction progress over time

### Legal Documentation
- **War Crimes Evidence**: Document deliberate targeting of civilian infrastructure
- **Insurance Claims**: Objective damage assessment for claims processing
- **Reconstruction Monitoring**: Verify rebuilding efforts

### Research & Analysis
- **Conflict Impact Studies**: Quantify destruction patterns
- **Urban Planning**: Inform resilient infrastructure design
- **Policy Development**: Evidence-based reconstruction policies

## Training Data Sources

The model was trained on diverse datasets including:
- Syria conflict imagery (2011-2024)
- Ukraine conflict imagery (2022-2024)
- Natural disaster imagery (earthquakes, floods, hurricanes)
- Validated damage assessments from humanitarian organizations

## Limitations

- **Temporal Dependency**: Requires both before and after images
- **Resolution Sensitivity**: Best performance with high-resolution imagery
- **Weather Conditions**: Accuracy affected by cloud cover, shadows
- **Damage Type**: Optimized for structural damage, not interior damage
- **Regional Variations**: Performance may vary by building types/regions

## Ethical Considerations

This model is designed for humanitarian and legal purposes:
- Document civilian harm for accountability
- Support humanitarian response efforts
- Aid in reconstruction planning
- Provide evidence for legal proceedings

**Not intended for**:
- Military targeting or operations
- Surveillance of civilian populations
- Discriminatory practices

## Integration Examples

### With GIS Systems
```python
import geopandas as gpd

def assess_region(shapefile_path, imagery_folder):
    gdf = gpd.read_file(shapefile_path)
    for idx, building in gdf.iterrows():
        before, after = get_building_imagery(building.geometry, imagery_folder)
        damage = assess_damage(before, after)
        gdf.at[idx, 'damage_level'] = damage
    return gdf
```

### Report Generation
```python
def generate_damage_report(assessments):
    summary = {
        'total_buildings': len(assessments),
        'no_damage': sum(1 for a in assessments if a['damage_level'] == 'No Damage'),
        'minor_damage': sum(1 for a in assessments if a['damage_level'] == 'Minor Damage'),
        'major_damage': sum(1 for a in assessments if a['damage_level'] == 'Major Damage'),
        'destroyed': sum(1 for a in assessments if a['damage_level'] == 'Destroyed')
    }
    return summary
```

## Citation

```bibtex
@model{building_damage_assessment_2024,
  title={BuildingDamage-Assessment: Temporal Building Damage Evaluation Model},
  author={Lemkin AI},
  year={2024},
  publisher={GitHub},
  url={https://github.com/LemkinAI/BuildingDamage-Assessment}
}
```

## License

MIT License - See [LICENSE](LICENSE) for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/LemkinAI/BuildingDamage-Assessment/issues)
- **Discussions**: [GitHub Discussions](https://github.com/LemkinAI/BuildingDamage-Assessment/discussions)
- **Email**: models@lemkinai.org

---
*Lemkin AI - Technology for Justice*