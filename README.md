# ComfyUI Inpaint Studio

A local web app that runs multiple ComfyUI inpainting pipelines in parallel with a user-drawn mask. After generation, YOLO hand detection runs automatically on each output.

## Prerequisites

- [Node.js](https://nodejs.org/) 18+
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running locally
- Python 3.9+ with `ultralytics` and `opencv-python` (for YOLO detection)

## Setup

### 1. Install Node.js dependencies

```bash
npm install
```

### 2. Configure environment

```bash
cp .env.local.example .env.local
```

Edit `.env.local` and set the paths for your machine — at minimum:

| Variable | Description |
|---|---|
| `COMFYUI_INPUT_DIR_WINDOWS` | ComfyUI input folder, e.g. `C:\Users\YOU\ComfyUI\input` |
| `YOLO_PYTHON` | Python executable with ultralytics installed |
| `YOLO_MODEL_PATH` | Path to your YOLO `.pt` weights file |

### 3. Set up YOLO Python environment

Create a dedicated conda (or venv) environment and install the required packages:

```bash
# With conda
conda create -n yolo python=3.9
conda activate yolo
pip install -r scripts/requirements.txt

# With venv
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r scripts/requirements.txt
```

Then point `YOLO_PYTHON` in `.env.local` to the Python executable of that environment:

```
# conda example
YOLO_PYTHON=C:\Users\YOU\miniconda3\envs\yolo\python.exe

# venv example
YOLO_PYTHON=C:\path\to\project\.venv\Scripts\python.exe
```

### 4. Place your ComfyUI workflows

- Export workflow JSON files from ComfyUI and place them in `./workflows/`.
- Keep `workflows/workflow-mapping.json` updated with the active workflow node IDs (or use the UI editor).

### 5. Start the app

```bash
npm run dev
```

## Usage

1. Set your ComfyUI URL in the **ComfyUI Connection** panel and click **Test Connection**.
2. Upload an image.
3. Paint the inpaint mask (white = inpaint area).
4. Tune parameters (seed, steps, CFG, sampler, scheduler, denoise, mask strength).
5. Click **Run** — results appear as they finish.

After each job completes, YOLO hand detection runs automatically and annotated bounding-box images are shown alongside each output.

## Data layout

| Path | Contents |
|---|---|
| `data/uploads/<jobId>/` | Original image + mask |
| `data/outputs/<jobId>/` | Generated output images (per workflow) |
| `data/yolo/<jobId>/` | Annotated images + `results.json` from YOLO |
| `data/jobs/` | Job metadata JSON files |

## Notes

- Workflow node IDs can be edited in the UI and are persisted to `workflows/workflow-mapping.json`.
- Each submitted job stores its own `comfyBaseUrl` snapshot, so changing the URL later does not affect already queued/running jobs.
- The app auto-detects `LoadImage`, `LoadImageMask`, and `KSampler` nodes if no mapping exists.
- YOLO runs as a background subprocess after the main inpainting job completes; a failed YOLO run will not affect the inpainting result.
