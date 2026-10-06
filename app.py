


import io
import os
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Drone Object Detection (VisDrone)", page_icon="🛰️", layout="wide")

WEIGHTS = "best.pt"
# Classes the model often confuses; merged so counts are more reliable.
MERGE = {"people": "pedestrian", "awning-tricycle": "tricycle"}


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    from ultralytics import YOLO
    return YOLO(WEIGHTS)


def detect(img_bytes, conf, merge, labels):
    model = load_model()
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
    rgb = np.array(img)
    bgr = np.ascontiguousarray(rgb[..., ::-1])  # Ultralytics expects BGR for numpy arrays
    r = model.predict(bgr, imgsz=960, conf=conf, max_det=1000,
                      agnostic_nms=True, verbose=False)[0]

    names = dict(r.names)
    if merge:
        names = {i: MERGE.get(n, n) for i, n in names.items()}
        r.names = names

    cls = r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) else []
    counts = Counter(names[int(c)] for c in cls)
    df = pd.DataFrame(sorted(counts.items(), key=lambda kv: -kv[1]), columns=["class", "count"])

    annotated = np.ascontiguousarray(r.plot(line_width=1, labels=labels, conf=labels)[..., ::-1])
    buf = io.BytesIO()
    Image.fromarray(annotated).save(buf, format="PNG")
    return {"original": rgb, "annotated": annotated, "counts": df,
            "total": int(len(cls)), "png": buf.getvalue(), "conf": conf}


tab_detect, tab_results = st.tabs(["Detect", "Data & Results"])

# ----------------------------------------------------------------- Detect tab
with tab_detect:
    st.title("Drone Object Detection (VisDrone)")
    st.caption("YOLO11s trained at 960 px on VisDrone with oversampling of rare classes. "
               "Detects: pedestrian, people, bicycle, car, van, truck, tricycle, "
               "awning-tricycle, bus, motor.")

    with st.form("detect_form"):
        up = st.file_uploader("Upload a drone image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        c1, c2 = st.columns([2, 1])
        conf = c1.slider("Confidence threshold", 0.05, 0.90, 0.30, 0.05,
                         help="Higher = fewer false boxes but more missed objects. 0.30 is the default.")
        merge = c2.checkbox("Merge similar classes", value=True,
                            help="people -> pedestrian, awning-tricycle -> tricycle")
        labels = c2.checkbox("Show labels on boxes", value=True)
        submitted = st.form_submit_button("Detect", type="primary")

    if submitted:
        if up is None:
            st.warning("Please upload an image first.")
        else:
            with st.spinner("Running detection (CPU, can take a few seconds)..."):
                st.session_state["result"] = detect(up.getvalue(), conf, merge, labels)

    res = st.session_state.get("result")
    if res:
        a, b = st.columns(2)
        a.image(res["original"], caption="Input")
        b.image(res["annotated"], caption=f"Detections: {res['total']} objects (confidence >= {res['conf']:.2f})")

        st.subheader("Objects found per class")
        if res["total"] == 0:
            st.info("Nothing detected. Try lowering the confidence threshold.")
        else:
            t1, t2 = st.columns(2)
            t1.dataframe(res["counts"], hide_index=True)
            t2.bar_chart(res["counts"].set_index("class")["count"])
        d1, d2 = st.columns(2)
        d1.download_button("Download annotated image", res["png"], "detections.png", "image/png")
        d2.download_button("Download counts (CSV)", res["counts"].to_csv(index=False),
                           "counts.csv", "text/csv")
    else:
        st.info("Upload an image and press **Detect**.")

# ------------------------------------------------------------ Data & Results tab
with tab_results:
    st.header("Data and results")

    st.subheader("Dataset: class imbalance")
    train_boxes = pd.Series({
        "pedestrian": 79337, "people": 27059, "bicycle": 10480, "car": 144867, "van": 24956,
        "truck": 12875, "tricycle": 4812, "awning-tricycle": 3246, "bus": 5926, "motor": 29647})
    st.bar_chart(train_boxes)
    st.write("Cars outnumber awning-tricycles about 44.6 to 1, and about 68% of all objects are "
             "smaller than 16 px at 640 px input. This is why the model uses 960 px input and "
             "oversampling of images with rare classes.")

    st.subheader("Model comparison")
    nan = float("nan")
    overall = pd.DataFrame([
        ["YOLO11n", 640, "none (baseline)", 0.263, 0.145, nan, nan],
        ["YOLO11n", 640, "oversampling", 0.274, 0.151, nan, nan],
        ["YOLO11s", 960, "oversampling", 0.473, 0.285, 0.391, 0.227],
    ], columns=["model", "imgsz", "imbalance strategy", "val mAP50", "val mAP50-95",
                "test mAP50", "test mAP50-95"])
    st.dataframe(overall, hide_index=True)
    st.caption("Validation was used to pick the weights, so the **test** score (VisDrone test-dev, "
               "1610 images) is the honest estimate.")

    st.subheader("Per-class results (final model)")
    per_class = pd.DataFrame([
        ["pedestrian", 0.555, 0.266, 0.368, 0.151], ["people", 0.421, 0.166, 0.196, 0.069],
        ["bicycle", 0.248, 0.113, 0.181, 0.078], ["car", 0.843, 0.598, 0.779, 0.496],
        ["van", 0.516, 0.372, 0.433, 0.293], ["truck", 0.441, 0.301, 0.466, 0.301],
        ["tricycle", 0.350, 0.200, 0.239, 0.129], ["awning-tricycle", 0.185, 0.113, 0.240, 0.147],
        ["bus", 0.616, 0.461, 0.612, 0.435], ["motor", 0.553, 0.261, 0.402, 0.169],
    ], columns=["class", "val AP50", "val AP50-95", "test AP50", "test AP50-95"])
    st.dataframe(per_class, hide_index=True)
    st.bar_chart(per_class.set_index("class")[["val AP50", "test AP50"]])

    st.subheader("Error analysis (validation set, confidence 0.25)")
    err = pd.DataFrame([
        ["pedestrian", 8844, 49.4, 4.4, 46.2], ["people", 5125, 36.9, 9.2, 53.9],
        ["bicycle", 1287, 25.3, 18.0, 56.7], ["car", 14064, 81.2, 2.4, 16.3],
        ["van", 1975, 44.8, 35.7, 19.5], ["truck", 750, 37.7, 24.3, 38.0],
        ["tricycle", 1045, 27.7, 23.0, 49.4], ["awning-tricycle", 532, 21.6, 30.1, 48.3],
        ["bus", 251, 52.2, 16.3, 31.5], ["motor", 4886, 48.3, 9.0, 42.7],
    ], columns=["class", "objects", "found %", "wrong class %", "missed %"])
    st.dataframe(err, hide_index=True)
    st.write("Most common confusions (true -> predicted): van -> car (662), people -> pedestrian (310), "
             "pedestrian -> people (304), car -> van (266), motor -> bicycle (168). "
             "Tiny objects (people, bicycles, pedestrians) are the ones most often missed entirely.")

    st.subheader("Confidence threshold trade-off (validation set)")
    sweep = pd.DataFrame([
        [0.15, 62.8, 14952, 27.3, 64.6], [0.25, 56.9, 7583, 13.8, 75.3],
        [0.30, 53.9, 5646, 10.3, 79.2], [0.35, 50.7, 4277, 7.8, 82.3],
        [0.40, 47.5, 3297, 6.0, 84.8], [0.50, 40.5, 1955, 3.6, 88.9],
    ], columns=["conf", "found %", "false boxes", "false per image", "precision %"])
    st.dataframe(sweep, hide_index=True)
    st.caption("0.30 is the default: beyond about 0.35, more real objects are lost than false boxes removed.")

    for fname, title in [("confusion_matrix_normalized.png", "Normalized confusion matrix"),
                         ("failure_examples.png", "Worst validation images")]:
        if os.path.exists(fname):
            st.subheader(title)
            st.image(fname)
