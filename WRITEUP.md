#  Write-up: Object Detection on Drone Imagery (VisDrone)

**Candidate:** Ruhab · **Model:** YOLO11s at 960 px · **Dashboard:** https://object-detection-on-drone-imagery-p9dheet3sdn7bmjxeytgq5.streamlit.app

## 1. Data findings
I used the full VisDrone DET data: 6,471 train, 548 val and 1,610 test-dev images, with ignored regions removed so only classes 0–9 are used.
- **Imbalance:** 144,867 cars against 3,246 awning-tricycles, a ratio of **44.6 : 1**. Pedestrians (79,337) and motors (29,647) are also common, while tricycles (4,812) and buses (5,926) are rare.
- **Tiny objects:** at 640 px input, about **68% of boxes are under 16 px** (33% under 8 px). The median pedestrian is 7.8 px wide.
- **Effect on training:** the model can ignore rare classes and still score well on cars, and at 640 px most objects shrink to a few pixels. I therefore used a higher input resolution and oversampling.

## 2. Imbalance method
- **Image-level oversampling** of images containing rare classes (6,471 → 8,138 training images, list in `train_oversampled.txt`, config in `dataset.yaml`).
- **Higher resolution:** 960 px instead of 640 px, so small objects keep more pixels.
- **Larger model:** YOLO11s instead of YOLO11n.

## 3. Results before and after
| Model | Img size | Strategy | Val mAP50 | Val mAP50-95 | Test mAP50 | Test mAP50-95 |
|---|---|---|---|---|---|---|
| YOLO11n | 640 | none (baseline) | 0.263 | 0.145 | n/a | n/a |
| YOLO11n | 640 | oversampling | 0.274 | 0.151 | n/a | n/a |
| **YOLO11s** | **960** | **oversampling** | **0.473** | **0.285** | **0.391** | **0.227** |

Oversampling alone gave a small gain (+0.011 mAP50). Most of the improvement came from the larger model and higher resolution. The test score is the honest estimate, since validation was used to choose the weights. Per-class AP, the confusion matrix and the confidence sweep are in the dashboard's *Data & Results* tab.

## 4. Failure analysis (validation, confidence 0.25)
The worst-case images are in `failure_examples.png`. The main failure patterns:
- **Vehicle confusion:** van → car (662 cases) and car → van (266). Vans and cars look alike from above.
- **Posture confusion:** people ↔ pedestrian (310 and 304). The two classes differ only by posture, which is hard to see at a few pixels.
- **Missed small objects:** 54% of people, 57% of bicycles and 46% of pedestrians are missed entirely, as are 49% of tricycles and 48% of awning-tricycles.
- **Motor ↔ bicycle** (168): a rider on a bike or motorbike is only a few pixels wide.
- **False boxes:** about 14 per image at 0.25, in crowded scenes (parking lots, markets) and on shadows or clutter. I set the default at 0.30 for the best balance.

## 5. What I would do with more time
- Train for many more epochs and with a larger model (YOLO11m/l).
- Add sliced (tiled) inference for small objects and compare it with normal inference.
- Improve preprocessing and augmentation, such as copy-paste for rare classes.
- Export to ONNX and report CPU speed.

**Not completed:**  the bonus items (tiled inference, ONNX export), and a second dataset.
