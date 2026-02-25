# ComfyUI Inpaint Studio

A local web app that runs multiple ComfyUI inpainting pipelines in parallel with a user-drawn mask.

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```
2. Place your ComfyUI workflow JSON exports in `./workflows`.
3. Update `workflows/workflow-mapping.json` with the node IDs for each workflow.
4. Ensure ComfyUI is running locally on `http://172.26.224.1:8188`. 
# (venv) (base) PS C:\Users\steckhan\ComfyUI> python main.py --listen 172.26.224.1 --port 8188

5. Confirm ComfyUI input directory matches `COMFYUI_INPUT_DIR_WINDOWS` in `lib/constants.ts`.
6. Start the app:
   ```bash
   npm run dev
   ```

## Usage

- Upload an image.
- Paint the inpaint mask (white = inpaint).
- Tune parameters (seed, steps, CFG, sampler, scheduler, denoise, mask strength).
- Run workflows (max 2 in parallel).

## Notes

- Output images are synced into `./data/outputs` after generation.
- Job metadata is stored in `./data/jobs`.
- Uploads and masks are stored in `./data/uploads`.
- Workflow node IDs can be edited in the UI and are persisted to `./workflows/workflow-mapping.json`.
- The app auto-detects `LoadImage`, `LoadImageMask`, and `KSampler` nodes if no mapping exists.
