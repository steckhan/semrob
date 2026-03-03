"use client";

import { useEffect, useRef, useState } from "react";

type MaskCanvasProps = {
  imageUrl: string | null;
  onMaskReady: (maskDataUrl: string) => void;
  /** Called whenever the mask changes, with a data URL of the source image
   *  composited with the painted mask as the alpha channel (RGBA PNG).
   *  Painted (white) areas become transparent, matching ComfyUI's LoadImage
   *  clipspace convention where transparent pixels mark the inpaint region. */
  onMaskedImageReady?: (rgbaDataUrl: string) => void;
};

export default function MaskCanvas({ imageUrl, onMaskReady, onMaskedImageReady }: MaskCanvasProps) {
  const displayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const isDrawingRef = useRef(false);
  const hasPaintedRef = useRef(false);
  const [brushSize, setBrushSize] = useState(30);
  const [brushOpacity, setBrushOpacity] = useState(0.8);
  const [overlayOpacity, setOverlayOpacity] = useState(0.35);
  const [mode, setMode] = useState<"draw" | "erase">("draw");
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    const stored = window.localStorage.getItem("maskOverlayOpacity");
    if (stored) {
      const value = Number(stored);
      if (!Number.isNaN(value)) {
        const clamped = Math.min(1, Math.max(0.05, value));
        setOverlayOpacity(clamped);
      }
    }
  }, []);

  const exportMask = () => {
    const maskCanvas = maskCanvasRef.current;
    if (!maskCanvas) {
      return null;
    }
    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = maskCanvas.width;
    exportCanvas.height = maskCanvas.height;
    const exportContext = exportCanvas.getContext("2d");
    if (!exportContext) {
      return null;
    }
    exportContext.fillStyle = "black";
    exportContext.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
    exportContext.drawImage(maskCanvas, 0, 0);
    return exportCanvas.toDataURL("image/png");
  };

  /**
   * Exports the source image composited with the painted mask as alpha.
   * Painted (white) pixels → transparent (alpha=0) in the output RGBA PNG.
   * This is the format ComfyUI's LoadImage expects for clipspace inpainting.
   */
  const exportMaskedImage = () => {
    const maskCanvas = maskCanvasRef.current;
    const image = imageRef.current;
    if (!maskCanvas || !image) {
      return null;
    }
    const w = maskCanvas.width;
    const h = maskCanvas.height;

    // Draw original image onto a temp canvas to read its pixels
    const srcCanvas = document.createElement("canvas");
    srcCanvas.width = w;
    srcCanvas.height = h;
    const srcCtx = srcCanvas.getContext("2d");
    if (!srcCtx) return null;
    srcCtx.drawImage(image, 0, 0, w, h);
    const srcData = srcCtx.getImageData(0, 0, w, h);

    // Read the mask pixels (white = painted region)
    const maskCtx = maskCanvas.getContext("2d");
    if (!maskCtx) return null;
    const maskData = maskCtx.getImageData(0, 0, w, h);

    // Compose RGBA: image RGB + inverted mask as alpha
    // Painted white areas (mask alpha > 0) → transparent in output
    const outCanvas = document.createElement("canvas");
    outCanvas.width = w;
    outCanvas.height = h;
    const outCtx = outCanvas.getContext("2d");
    if (!outCtx) return null;
    const outData = outCtx.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      const p = i * 4;
      outData.data[p] = srcData.data[p];         // R
      outData.data[p + 1] = srcData.data[p + 1]; // G
      outData.data[p + 2] = srcData.data[p + 2]; // B
      // Mask alpha channel: painted white → 0 (transparent = inpaint), unpainted → 255 (opaque = keep)
      outData.data[p + 3] = 255 - maskData.data[p + 3];
    }
    outCtx.putImageData(outData, 0, 0);
    return outCanvas.toDataURL("image/png");
  };

  useEffect(() => {
    window.localStorage.setItem(
      "maskOverlayOpacity",
      overlayOpacity.toString(),
    );
    if (isReady) {
      redrawOverlay();
      redrawDisplay();
    }
  }, [overlayOpacity, isReady]);

  useEffect(() => {
    if (
      !imageUrl ||
      !maskCanvasRef.current ||
      !overlayCanvasRef.current ||
      !displayCanvasRef.current
    ) {
      return;
    }

    setIsReady(false);
    const image = new Image();
    imageRef.current = image;
    image.src = imageUrl;
    image.onload = () => {
      const maskCanvas = maskCanvasRef.current;
      const overlayCanvas = overlayCanvasRef.current;
      const displayCanvas = displayCanvasRef.current;
      if (!maskCanvas || !overlayCanvas || !displayCanvas) {
        return;
      }
      maskCanvas.width = image.width;
      maskCanvas.height = image.height;
      overlayCanvas.width = image.width;
      overlayCanvas.height = image.height;
      displayCanvas.width = image.width;
      displayCanvas.height = image.height;

      const maskContext = maskCanvas.getContext("2d");
      const overlayContext = overlayCanvas.getContext("2d");
      const displayContext = displayCanvas.getContext("2d");
      if (!maskContext || !overlayContext || !displayContext) {
        return;
      }
      maskContext.clearRect(0, 0, maskCanvas.width, maskCanvas.height);

      overlayContext.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

      displayContext.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
      displayContext.drawImage(image, 0, 0);

      hasPaintedRef.current = false;
      setIsReady(true);
      redrawOverlay();
      redrawDisplay();

      const dataUrl = exportMask();
      if (dataUrl) {
        onMaskReady(dataUrl);
      }
      const rgbaUrl = exportMaskedImage();
      if (rgbaUrl) {
        onMaskedImageReady?.(rgbaUrl);
      }
    };
  }, [imageUrl, onMaskReady, onMaskedImageReady]);

  const startDrawing = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = displayCanvasRef.current;
    if (!canvas) {
      return;
    }
    canvas.setPointerCapture(event.pointerId);
    isDrawingRef.current = true;
    setIsDrawing(true);
    draw(event);
  };

  const stopDrawing = (event?: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = displayCanvasRef.current;
    if (canvas && event) {
      canvas.releasePointerCapture(event.pointerId);
    }
    isDrawingRef.current = false;
    setIsDrawing(false);
    const dataUrl = exportMask();
    if (dataUrl) {
      onMaskReady(dataUrl);
    }
    const rgbaUrl = exportMaskedImage();
    if (rgbaUrl) {
      onMaskedImageReady?.(rgbaUrl);
    }
  };

  const redrawOverlay = () => {
    const maskCanvas = maskCanvasRef.current;
    const overlayCanvas = overlayCanvasRef.current;
    if (!maskCanvas || !overlayCanvas || !isReady) {
      return;
    }
    const overlayContext = overlayCanvas.getContext("2d");
    if (!overlayContext) {
      return;
    }
    overlayContext.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    if (!hasPaintedRef.current) {
      return;
    }
    overlayContext.globalCompositeOperation = "source-over";
    overlayContext.drawImage(maskCanvas, 0, 0);
    overlayContext.globalCompositeOperation = "source-in";
    overlayContext.fillStyle = `rgba(239,68,68,${overlayOpacity})`;
    overlayContext.fillRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  };

  const redrawDisplay = () => {
    const displayCanvas = displayCanvasRef.current;
    const overlayCanvas = overlayCanvasRef.current;
    const image = imageRef.current;
    if (!displayCanvas || !overlayCanvas || !image || !isReady) {
      return;
    }
    const displayContext = displayCanvas.getContext("2d");
    if (!displayContext) {
      return;
    }
    displayContext.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
    displayContext.drawImage(image, 0, 0);
    displayContext.drawImage(overlayCanvas, 0, 0);
  };

  const draw = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const maskCanvas = maskCanvasRef.current;
    const overlayCanvas = overlayCanvasRef.current;
    const displayCanvas = displayCanvasRef.current;
    if (!maskCanvas || !overlayCanvas || !displayCanvas || !isReady) {
      return;
    }
    if (!isDrawingRef.current) {
      return;
    }
    const rect = displayCanvas.getBoundingClientRect();
    const maskContext = maskCanvas.getContext("2d");
    const overlayContext = overlayCanvas.getContext("2d");
    if (!maskContext || !overlayContext) {
      return;
    }
    const scaleX = displayCanvas.width / rect.width;
    const scaleY = displayCanvas.height / rect.height;
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;

    const composite = mode === "draw" ? "source-over" : "destination-out";
    maskContext.globalCompositeOperation = composite;
    overlayContext.globalCompositeOperation = composite;

    maskContext.fillStyle = `rgba(255,255,255,${brushOpacity})`;

    maskContext.beginPath();
    maskContext.arc(x, y, brushSize, 0, Math.PI * 2);
    maskContext.fill();

    hasPaintedRef.current = true;
    redrawOverlay();
    redrawDisplay();
  };

  const clearMask = () => {
    const maskCanvas = maskCanvasRef.current;
    const overlayCanvas = overlayCanvasRef.current;
    const displayCanvas = displayCanvasRef.current;
    const image = imageRef.current;
    if (!maskCanvas || !overlayCanvas || !displayCanvas || !image || !isReady) {
      return;
    }
    const maskContext = maskCanvas.getContext("2d");
    const overlayContext = overlayCanvas.getContext("2d");
    const displayContext = displayCanvas.getContext("2d");
    if (!maskContext || !overlayContext || !displayContext) {
      return;
    }
    maskContext.globalCompositeOperation = "source-over";
    maskContext.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    hasPaintedRef.current = false;

    overlayContext.globalCompositeOperation = "source-over";
    overlayContext.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    displayContext.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
    displayContext.drawImage(image, 0, 0);

    redrawOverlay();
    redrawDisplay();

    const dataUrl = exportMask();
    if (dataUrl) {
      onMaskReady(dataUrl);
    }
    const rgbaUrl = exportMaskedImage();
    if (rgbaUrl) {
      onMaskedImageReady?.(rgbaUrl);
    }
  };

  return (
    <div className="panel">
      <div className="row">
        <div style={{ flex: 1 }}>
          <label htmlFor="brush">Brush Size</label>
          <input
            id="brush"
            className="input"
            type="range"
            min={4}
            max={80}
            value={brushSize}
            onChange={(event) => setBrushSize(Number(event.target.value))}
          />
        </div>
        <div style={{ flex: 1 }}>
          <label htmlFor="opacity">Mask Opacity</label>
          <input
            id="opacity"
            className="input"
            type="range"
            min={0.1}
            max={1}
            step={0.05}
            value={brushOpacity}
            onChange={(event) => setBrushOpacity(Number(event.target.value))}
          />
        </div>
        <div style={{ flex: 1 }}>
          <label htmlFor="overlay">Overlay Opacity</label>
          <input
            id="overlay"
            className="input"
            type="range"
            min={0.05}
            max={1}
            step={0.05}
            value={overlayOpacity}
            onChange={(event) => setOverlayOpacity(Number(event.target.value))}
          />
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "flex-end" }}>
          <button
            type="button"
            className="button"
            onClick={() => setMode(mode === "draw" ? "erase" : "draw")}
          >
            Mode: {mode === "draw" ? "Paint" : "Erase"}
          </button>
          <button type="button" className="button" onClick={clearMask}>
            Clear
          </button>
        </div>
      </div>
      <div className="canvas-wrapper">
        {imageUrl ? (
          <>
            <canvas ref={maskCanvasRef} className="canvas-hidden" />
            <canvas ref={overlayCanvasRef} className="canvas-hidden" />
            <canvas
              ref={displayCanvasRef}
              className="canvas-display"
              onPointerDown={startDrawing}
              onPointerUp={stopDrawing}
              onPointerLeave={stopDrawing}
              onPointerCancel={stopDrawing}
              onPointerMove={draw}
              style={{ touchAction: "none", cursor: "crosshair" }}
            />
          </>
        ) : (
          <div className="small">Upload an image to start masking.</div>
        )}
      </div>
      <div className="small">
        Red overlay = inpaint region (mask saved as white).
      </div>
    </div>
  );
}
