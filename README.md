# SemProbe — Semantic Robustness Probing via Controlled Inpainting

> Code repository for the **ECML 2025 Demo Track** paper
> *"Semantic Robustness Probing via Controlled Inpainting: An Interactive Tool for Safety-Critical Object Detection"*
> Nico Steckhan, Krutarth Prajapati, Weija Shao, Silvia Vock · BAuA, Germany

SemProbe is a local tool for systematically stress-testing object detectors against semantically meaningful, deployment-realistic image variations. With the EU AI Act classifying camera-based safety functions as high-risk AI, standard domain-agnostic benchmarks are no longer sufficient — robustness must be evidenced against factors derived from the actual **Operational Design Domain (ODD)**.

The workflow: upload a deployment image, draw or auto-generate a mask over the region of interest, apply a controlled inpainting modification driven by an ODD factor catalog (e.g. *"cut-resistant work glove"*, *"heavy sawdust"*, *"specular glare"*), and immediately compare YOLO detection confidence and bounding boxes before and after. Every probe is logged with its factor, level, prompt, and confidence delta — exportable as CSV/JSON aligned with ISO/IEC TR 24029-1 and EU AI Act documentation requirements.

Inpainting runs entirely locally on consumer GPUs using [FLUX.2-klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (4B, ~13 GB VRAM) or the larger [FLUX.2-klein 9B fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) (~18 GB VRAM), preserving data sovereignty for safety-critical industrial imagery.

**Key capabilities:**

- **ODD-guided probing** — structure modifications along four ODD dimensions (actors, activities, environment, sensors) per ISO 34503 / BSI PAS 1883; catalogs can be authored manually or derived semi-automatically via LLM
- **Manual & SAM2 auto-mask** — draw a mask by hand or let GroundingDINO + SAM2 auto-segment the target object from a text prompt
- **Batch processing** — queue a full factor catalog across multiple images with live progress tracking and per-image mask overlays
- **Parallel variations** — generate multiple seeds or inpainting variants simultaneously for robustness aggregation
- **Side-by-side YOLO comparison** — detection results on original and modified images shown together with bounding boxes, confidence scores, and delta
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
| Diffusion model (4B, ~13 GB VRAM) | FLUX.2-klein 4B | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) |
| Diffusion model (9B fp8, ~18 GB VRAM) | FLUX.2-klein 9B fp8 | [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) |
| Text encoder for 4B | Qwen3 4B (split files) | [Comfy-Org · text\_encoders](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b/tree/main/split_files/text_encoders) |
| Text encoder for 9B | Qwen3 8B fp8mixed | [Comfy-Org · text\_encoders 9B](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/tree/main/split_files/text_encoders) |
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
| `GT_DIR` | *(optional)* Path to a folder of YOLO-format `.txt` ground-truth label files. When set and a matching label file exists for an uploaded image, per-frame AP/precision/recall/F1 and accumulated batch metrics are computed automatically. |

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
- **Model selection** — the **ComfyUI Connection** card exposes a segmented control to switch between *Flux.2 Klein 4B* and *Flux.2 Klein 9B (fp8)*. Selecting the 9B model automatically patches both the UNETLoader (`flux-2-klein-9b-fp8.safetensors`) and the CLIPLoader (`qwen_3_8b_fp8mixed.safetensors`) in the workflow at submission time — no manual workflow editing required. The choice is persisted in `localStorage`.
