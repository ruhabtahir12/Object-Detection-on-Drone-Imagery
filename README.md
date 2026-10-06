# Object Detection on Drone Imagery (VisDrone)

Detects 10 object classes in aerial drone images using **YOLO11s** trained at **960 px**, with oversampling of rare classes. Includes an interactive Streamlit dashboard for running detection and exploring the results.

### Live dashboard: **[object-detection-on-drone-imagery-p9dheet3sdn7bmjxeytgq5.streamlit.app](https://object-detection-on-drone-imagery-p9dheet3sdn7bmjxeytgq5.streamlit.app)**

> Upload a drone image, press **Detect**, and get an annotated image plus per-class object counts. The second tab shows the dataset analysis, model comparison, per-class scores, error analysis and confidence-threshold trade-off.

---

## Classes

`pedestrian`, `people`, `bicycle`, `car`, `van`, `truck`, `tricycle`, `awning-tricycle`, `bus`, `motor`

## Dataset

[VisDrone](https://github.com/VisDrone/VisDrone-Dataset) (via Ultralytics `VisDrone.yaml`).

| Split | Images |
|---|---|
| Train | 6,471 (8,138 after oversampling) |
| Val | 548 |
| Test (test-dev) | 1,610 |

Two properties of the data drive the design:

- **Heavy class imbalance:** 144,867 cars vs 3,246 awning-tricycles (**44.6 : 1**).
- **Tiny objects:** about **68%** of boxes are under 16 px at 640 px input (median pedestrian ≈ 7.8 px).

So the final model uses a higher input resolution (960 px) and oversamples images that contain rare classes.

## Approach

| Step | Details |
|---|---|
| Baseline | YOLO11n, 640 px, 15 epochs, no imbalance handling |
| Oversampling | Same model, images with rare classes repeated in training |
| Final model | YOLO11s, 960 px, oversampling, 15 epochs plus a 10-epoch continuation |
| Inference | `imgsz=960`, `max_det=1000`, class-agnostic NMS, default confidence 0.30 |

## Results

| Model | Img size | Imbalance strategy | Val mAP50 | Val mAP50-95 | Test mAP50 | Test mAP50-95 |
|---|---|---|---|---|---|---|
| YOLO11n | 640 | none (baseline) | 0.263 | 0.145 | n/a | n/a |
| YOLO11n | 640 | oversampling | 0.274 | 0.151 | n/a | n/a |
| **YOLO11s** | **960** | **oversampling** | **0.473** | **0.285** | **0.391** | **0.227** |

Weights were selected on the validation set, so the **test** score (0.391 mAP50) is the honest estimate of performance.

### Per-class AP50 (final model)

| Class | Val | Test |
|---|---|---|
| car | 0.843 | 0.779 |
| bus | 0.616 | 0.612 |
| pedestrian | 0.555 | 0.368 |
| motor | 0.553 | 0.402 |
| van | 0.516 | 0.433 |
| truck | 0.441 | 0.466 |
| people | 0.421 | 0.196 |
| tricycle | 0.350 | 0.239 |
| bicycle | 0.248 | 0.181 |
| awning-tricycle | 0.185 | 0.240 |

### Confidence threshold trade-off (validation)

| Conf | Objects found | False boxes / image | Precision |
|---|---|---|---|
| 0.15 | 62.8% | 27.3 | 64.6% |
| 0.25 | 56.9% | 13.8 | 75.3% |
| **0.30** | **53.9%** | **10.3** | **79.2%** |
| 0.35 | 50.7% | 7.8 | 82.3% |
| 0.50 | 40.5% | 3.6 | 88.9% |

0.30 is the dashboard default; beyond ~0.35 more real objects are lost than false boxes are removed.

## Limitations

- **Modest overall accuracy.** Test mAP50 is 0.391 (mAP50-95 0.227). Even at the default threshold only about 54% of objects are found on validation, so this is not suitable for exact counting or safety-critical use.
- **Tiny objects are often missed.** With ~68% of objects under 16 px, small classes suffer most: only 25–37% of people, bicycles, tricycles and awning-tricycles are found, and 43–57% of people, bicycles and pedestrians are missed entirely.
- **Rare classes remain weak.** Oversampling helped, but awning-tricycle (AP50 0.185 val), bicycle (0.248) and tricycle (0.350) are still far behind car (0.843). The gain from oversampling alone was small (mAP50 0.263 → 0.274); most of the improvement came from the larger model and higher resolution.
- **Confusable classes.** Common errors: van → car (662), people → pedestrian (310), pedestrian → people (304), car → van (266), motor → bicycle (168). The dashboard's "Merge similar classes" option (people → pedestrian, awning-tricycle → tricycle) hides some of these, but it is a workaround, not a fix. Merged counts no longer reflect the original 10 labels.
- **Precision/recall trade-off.** At 0.30 about 10 false boxes per image appear on validation. Raising the threshold removes them but misses more real objects.
- **Validation–test gap.** Scores drop from validation to test (0.473 → 0.391 mAP50), and some classes fall sharply (people 0.421 → 0.196, pedestrian 0.555 → 0.368), suggesting some overfitting to the validation set or a domain shift.
- **Short training.** The model was trained for about 25 epochs on free-tier GPUs (Colab/Kaggle). Longer training, a larger model (YOLO11m/l) or tiled/sliced inference would likely improve small-object detection.
- **Dataset domain.** VisDrone was captured in specific Chinese cities, at specific altitudes and camera angles. Performance may drop on other regions, altitudes, night scenes or weather conditions.
- **Slow on CPU.** The hosted dashboard runs inference on CPU at 960 px, so large images can take several seconds. A free-tier host may also sleep when idle, so the first load can be slow.
- **Counts are detections, not ground truth.** Per-class counts shown in the dashboard are model outputs and inherit all the errors above.

## Run locally

```bash
git clone https://github.com/ruhabtahir12/Object-Detection-on-Drone-Imagery.git
cd Object-Detection-on-Drone-Imagery
pip install -r requirements.txt
streamlit run app.py
```

`best.pt` (the trained weights) must sit next to `app.py`. `packages.txt` lists system packages needed on Streamlit Community Cloud.

## Repository contents

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard (Detect tab and Data & Results tab) |
| `best.pt` | Final YOLO11s weights |
| `Smaller-model-training-notebook.ipynb` | EDA, YOLO11n baseline, oversampling, YOLO11s first run (Colab) |
| `larger-model-training-notebook.ipynb` | YOLO11s 960 px continuation, val/test evaluation, error analysis, threshold sweep (Kaggle) |
| `confusion_matrix_normalized.png` | Normalized confusion matrix (shown in dashboard) |
| `requirements.txt`, `packages.txt` | Python and system dependencies |

## Tech stack

Python · Ultralytics YOLO11 · PyTorch · Streamlit · pandas · NumPy · Pillow

## Acknowledgements

- [VisDrone dataset](https://github.com/VisDrone/VisDrone-Dataset) (Tianjin University, AISKYEYE team)
- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
