import fs from "fs";
import os from "os";
import path from "path";

const ENV_ROOT = process.env.INPAINT_ROOT;

function resolveProjectRoot(): string {
  if (ENV_ROOT && fs.existsSync(path.join(ENV_ROOT, "workflows"))) {
    return ENV_ROOT;
  }

  let current = process.cwd();
  for (let depth = 0; depth < 6; depth += 1) {
    if (fs.existsSync(path.join(current, "workflows"))) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) {
      break;
    }
    current = parent;
  }

  const fallback = "C:\\dev\\inpaintntest";
  if (fs.existsSync(path.join(fallback, "workflows"))) {
    return fallback;
  }

  return process.cwd();
}

const PROJECT_ROOT = resolveProjectRoot();

function normalizeBaseUrl(value: string): string {
  return value.replace(/\/+$/, "");
}

export const COMFYUI_BASE_URL = normalizeBaseUrl(
  process.env.COMFYUI_BASE_URL ?? "http://localhost:8188",
);
export const OPENAI_API_KEY = process.env.OPENAI_API_KEY ?? "";
export const MAX_PARALLEL_WORKFLOWS = 2;
export const DATA_ROOT = path.join(PROJECT_ROOT, "data");
export const WORKFLOWS_DIR = path.join(PROJECT_ROOT, "workflows");
const _winUser = process.env.USERNAME ?? process.env.USER ?? os.userInfo().username;
export const COMFYUI_INPUT_DIR =
  process.env.COMFYUI_INPUT_DIR ?? `/mnt/c/Users/${_winUser}/ComfyUI/input`;
export const COMFYUI_INPUT_DIR_WINDOWS =
  process.env.COMFYUI_INPUT_DIR_WINDOWS ?? `C:\\Users\\${_winUser}\\ComfyUI\\input`;
export const COMFYUI_OUTPUT_DIR =
  process.env.COMFYUI_OUTPUT_DIR ?? `/mnt/c/Users/${_winUser}/ComfyUI/output`;
export const COMFYUI_OUTPUT_DIR_WINDOWS =
  process.env.COMFYUI_OUTPUT_DIR_WINDOWS ?? `C:\\Users\\${_winUser}\\ComfyUI\\output`;
export const YOLO_PYTHON =
  process.env.YOLO_PYTHON ?? "python";
export const YOLO_CONF_THRESHOLD = 0.25;
export const YOLO_SCRIPT_PATH =
  process.env.YOLO_SCRIPT_PATH ??
  path.join(PROJECT_ROOT, "scripts", "yolo_detect.py");
export const YOLO_MODEL_PATH =
  process.env.YOLO_MODEL_PATH ??
  path.join(PROJECT_ROOT, "model", "best.pt");
export const GT_DIR =
  process.env.GT_DIR ??
  path.join(PROJECT_ROOT, "IPA_repo", "obj_Test_data");
