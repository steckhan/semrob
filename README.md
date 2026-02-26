# ComfyUI Inpaint Studio

A local web app that runs multiple ComfyUI inpainting pipelines in parallel with a user-drawn mask.

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```
2. Place your ComfyUI workflow JSON exports in `./workflows`.
3. Keep `workflows/workflow-mapping.json` updated with the active workflow node IDs.
4. Confirm ComfyUI input/output directories in `lib/constants.ts` match your local ComfyUI setup.
5. Optionally set a default ComfyUI URL for your environment:
   ```bash
   # .env.local
   COMFYUI_BASE_URL=http://127.0.0.1:8188
   ```
6. Start the app:
   ```bash
   npm run dev
   ```

## Usage

- Set your ComfyUI URL in the **ComfyUI Connection** panel.
- Click **Test Connection** to verify `/system_stats` is reachable.
- Upload an image.
- Paint the inpaint mask (white = inpaint).
- Tune parameters (seed, steps, CFG, sampler, scheduler, denoise, mask strength).
  - Current defaults: sampler `euler`, scheduler `normal`, denoise `1`.
- Run workflows (max 2 in parallel).

## Notes

- Output images are synced into `./data/outputs` after generation.
- Job metadata is stored in `./data/jobs`.
- Uploads and masks are stored in `./data/uploads`.
- Workflow node IDs can be edited in the UI and are persisted to `./workflows/workflow-mapping.json`.
- Each submitted job stores its own `comfyBaseUrl` snapshot, so changing the URL later does not affect already queued/running jobs.
- The app auto-detects `LoadImage`, `LoadImageMask`, and `KSampler` nodes if no mapping exists.
