# 🅿️ Parking Free‑Space Detection (Radar + Camera)

Lightweight pipeline for **free‑space segmentation** in parking scenes, built for **ADAS** constraints. We fuse **radar** and **camera** in a common **ego‑frame BEV grid** using **motion cancellation + IPM** for camera and a direct (x,y) mapping for radar. Designed to start simple (minimal compute/memory) and expand later to stronger BEV techniques.

---

## ✨ Why this repo
- **Immediate goal:** Prove out *parking free‑space* detection.  
- **Practical reframing:** Until we get parking‑specific data from the customer, we train on **open‑source driving data** (nuScenes) as a harder proxy task. Parking scenes should be simpler once data arrives.
- **Hardware reality:** ADAS targets have tight runtime/memory budgets → begin with **compute‑light** methods, then iterate.

---

## 📦 Dataset & Modalities
- **Dataset:** nuScenes (camera + radar).
- **Modalities:** 
  - **Camera** → BEV via **motion cancellation + IPM** (Inverse Perspective Mapping) using calibration.
  - **Radar** → BEV via dropping z and rasterizing (x,y) in ego frame.
- **Common representation:** **BEV grid in ego frame** (x forward, y left, z up).

> We evaluated *what to fuse, how to fuse, when to fuse* via survey literature. In this POC we chose **early fusion** by channel‑wise concatenation of BEV maps.

---

## 🧭 Coordinate Frames (nuScenes)
- **Ego frame:** x forward (longitudinal), y left (lateral), z up.  
- **Global → Ego:** use nuScenes `ego_pose` (translation + rotation quaternion).  
- **Sensor → Ego:** use `calibrated_sensor` (translation + rotation quaternion).

---

## 🧱 BEV Grid Definition
- **ROI (meters):**
  - Longitudinal (x): e.g., `[0, 50]`
  - Lateral (y): e.g., `[-15, 15]`
- **Resolution:** e.g., `0.20 m` per cell (tune as needed)
- **BEV size:** `H = (x_max - x_min)/res`, `W = (y_max - y_min)/res`
- **Channels:** at minimum `1 (radar) + 3 (camera)`; can extend.

---

## 📡 Radar → BEV (Straightforward)
1. Load radar point cloud for the chosen sensor(s).
2. Transform **sensor → ego** using `calibrated_sensor`.
3. **Drop z**, keep (x,y) within BEV ROI.
4. Rasterize to grid (e.g., occupancy=1 per cell with any radar point). Optionally store extra channels (RCS, velocity) later.

**Notes**
- Multi‑sensor: merge front/left/right radars (transform each to ego then rasterize).
- Filtering: RCS threshold, max range, Doppler sanity checks (optional, configurable).

---

## 📷 Camera → BEV via Motion Cancellation + IPM (Compute‑Light)
**Why:** Monocular depth is expensive/ambiguous; transformers are heavier. For a POC on ADAS targets, we begin with geometry.

**Steps**
1. **Ego motion compensation**: Stabilize the camera view (reduce roll/pitch/yaw/translation effects) between frames using ego poses, or operate per‑frame with the correct pose at capture time (choose one consistent approach).
2. **Get camera intrinsics/extrinsics:** from nuScenes `calibrated_sensor`.
3. **Define ground plane assumption** (flat road approximation) or use map/IMU‑derived pitch/roll to refine.
4. **IPM warp:** Map pixels that correspond to the ground plane to (x,y) on the BEV grid.
5. **Camera BEV raster:** Produce a binary/continuous layer (e.g., “visible ground likelihood” or “free‑space prior”).

**Known limitations**
- Flat‑ground assumption; non‑planar scenes may distort.
- Objects taller than ground plane (cars, curbs) need masking (e.g., using radar occupancy or semantics) to avoid projecting them as free space.

---

## 🔗 Fusion Strategy (What / How / When)
- **What:** Camera‑BEV + Radar‑BEV (ego grid).
- **How:** **Early fusion** → channel‑wise concatenation.
- **When:** Pre‑network; feed the stacked tensor to the model.

Future options: mid‑level fusion (feature maps) or late fusion (ensembles/post‑merge) if needed.

---

## 🎯 Ground‑Truth (GT) Generation
- Source: `sample_annotation` boxes.  
- Convert annotation boxes **global → ego** at the image/radar timestamp.  
- **Rasterize occupancy**: fill cells overlapped by any annotated object footprint (polygons from 3D boxes projected to ground plane).  
- Handle **clipping** at BEV boundaries; partial overlaps count as occupied.

This yields an **occupied mask**; **free space** is simply `~occupied` within ROI (optionally erode/dilate to align with grid resolution).

---

## 🧪 Models & Training
- **Task:** segmentation in BEV (binary: free vs occupied).  
- **Input:** `C×H×W` BEV tensor; `C≥4` (radar, camera).  
- **Loss:** e.g., BCE + Dice; metrics: **IoU, Dice**.  
- **Data splits:** follow nuScenes scenes or frame‑level splits; avoid leakage across adjacent frames.

**Baselines**
- Lightweight UNet/DeepLabv3 head on BEV tensors.
- Optionally add shallow CNN for speed.

---

## 🧰 Installation
```bash
# Python == 3.10 recommended
python -m venv nuscenes_dev && source nuscenes_dev/bin/activate  # (Windows: .venv\Scripts\activate)
pip install .
pip install ".[dev]"
```

## 🗺️ Roadmap (next iterations)
- **Camera BEV:** try monocular depth (MiDaS or similar) and re‑project to BEV.
- **Transformer BEV:** direct image‑to‑BEV encoders.
- **Fusion:** mid‑level/late fusion, uncertainty‑aware fusion, radar velocity/RCS channels.
- **Better GT:** use map layers when available; refine static vs dynamic occupancy.
- **Robustness:** curb/terrain awareness, occlusion handling.
- **Metrics:** add PR curves for free/occupied, per‑range IoU, latency/throughput.

---

## ⚠️ Caveats
- IPM assumes a reasonably flat ground; expect distortions on slopes/ramps.
- Time alignment across sensors matters; ensure the correct `ego_pose` at capture time.
- nuScenes coordinate conventions must be respected end‑to‑end.

---

## 📚 Papers Referenced
- **(a) Multimodal fusion survey:** Baltrušaitis et al., *Multimodal Machine Learning: A Survey and Taxonomy*. arXiv:1902.07830.  
  PDF: https://arxiv.org/pdf/1902.07830
- **(b) BEV of camera survey:** Ma et al., *Vision-Centric BEV Perception: A Survey*. arXiv:2208.02797.  
  PDF: https://arxiv.org/pdf/2208.02797
- **(c) Vehicle Position & Orientation via IPM:** Kim et al., *Deep Learning‑based Vehicle Position and Orientation Estimation via Inverse Perspective Mapping Image*.  
  PDF: https://www.researchgate.net/profile/Youngseok-Kim-9/publication/335499200_Deep_Learning_based_Vehicle_Position_and_Orientation_Estimation_via_Inverse_Perspective_Mapping_Image/links/5efeb41a45851550508795c3/Deep-Learning-based-Vehicle-Position-and-Orientation-Estimation-via-Inverse-Perspective-Mapping-Image.pdf
