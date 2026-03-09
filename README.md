# ComfyUI Inpaint Studio

A local web UI for AI-powered inpainting built on top of [ComfyUI](https://github.com/comfyanonymous/ComfyUI). Paint a mask over any area of an image, tune generation parameters, and run the FLUX.2-klein diffusion model — either in single-image or batch mode.

**Key features:**

- **Manual & SAM2 auto-mask** — draw a mask by hand or let GroundingDINO + SAM2 detect and segment the target object automatically from a text prompt
- **Batch processing** — queue multiple images at once, with per-image mask overlays and live progress tracking
- **Parallel variations** — generate multiple seeds or workflow variants simultaneously
- **YOLO detection** — automatic hand/object detection runs on every output and annotated results are shown alongside the generated images
- **Workflow editor** — map ComfyUI node IDs to parameters directly in the UI without touching JSON

## Prerequisites

- [Node.js](https://nodejs.org/) 18+
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running locally
- Python 3.9+ with `ultralytics` and `opencv-python` (for YOLO detection)

## ComfyUI Setup

### Model Weights

Download and place these into your ComfyUI `models/` subdirectories:

| Role | Model | Download |
|---|---|---|
| Diffusion model | FLUX.2-klein 4B | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) |
| Text encoder | Qwen3 4B (split files) | [Comfy-Org · text\_encoders](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b/tree/main/split_files/text_encoders) |
| VAE | FLUX.2 VAE (split files) | [Comfy-Org · vae](https://huggingface.co/Comfy-Org/flux2-dev/tree/main/split_files/vae) |
| Segmentation | SAM2 Hiera Base Plus | [facebook/sam2-hiera-base-plus](https://huggingface.co/facebook/sam2-hiera-base-plus) |
| Object detection | GroundingDINO SwinB | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base) |

### Custom Nodes

Install the following via **ComfyUI Manager**:

- ComfyUI Impact Pack
- ComfyUI-SAM2
- ComfyUI LayerStyle
- rgthree-comfy
- ComfyUI Easy Use
- ComfyUI KJNodes
- ComfyUI_essentials
- WAS Node Suite

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

Install the required packages into a Python 3.9+ environment:

```bash
# Option A — reuse an existing conda env (e.g. your GPU/ML env):
C:\Users\YOU\miniconda3\envs\gpu\python.exe -m pip install ultralytics opencv-python

# Option B — create a fresh dedicated env:
conda create -n yolo python=3.9
conda activate yolo
pip install -r scripts/requirements.txt
```

Then point `YOLO_PYTHON` in `.env.local` to that Python executable:

```
# existing gpu env
YOLO_PYTHON=C:\Users\YOU\miniconda3\envs\gpu\python.exe

# or a fresh yolo env
YOLO_PYTHON=C:\Users\YOU\miniconda3\envs\yolo\python.exe
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
