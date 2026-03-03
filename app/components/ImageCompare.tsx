"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Props = {
  originalUrl: string;
  inpaintedUrl: string;
  onClose: () => void;
};

export default function ImageCompare({ originalUrl, inpaintedUrl, onClose }: Props) {
  const [sliderPos, setSliderPos] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const moveTo = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const pct = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
    setSliderPos(pct);
  }, []);

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (dragging.current) moveTo(e.clientX);
    };
    const onTouchMove = (e: TouchEvent) => {
      if (dragging.current) moveTo(e.touches[0].clientX);
    };
    const onUp = () => {
      dragging.current = false;
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("touchmove", onTouchMove, { passive: true });
    window.addEventListener("mouseup", onUp);
    window.addEventListener("touchend", onUp);
    window.addEventListener("keydown", onKey);

    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("touchmove", onTouchMove);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("touchend", onUp);
      window.removeEventListener("keydown", onKey);
    };
  }, [moveTo, onClose]);

  const startDrag = (clientX: number) => {
    dragging.current = true;
    moveTo(clientX);
  };

  return (
    <div className="compare-overlay" onClick={onClose}>
      <div className="compare-modal" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="compare-header">
          <span>Before / After</span>
          <button className="btn btn-outline btn-sm" onClick={onClose}>
            ✕ Close
          </button>
        </div>

        {/* Slider area */}
        <div
          ref={containerRef}
          className="compare-container"
          onMouseDown={(e) => startDrag(e.clientX)}
          onTouchStart={(e) => startDrag(e.touches[0].clientX)}
        >
          {/* Original — sets the natural container height */}
          <img
            src={originalUrl}
            alt="Original"
            className="compare-img"
            draggable={false}
          />

          {/* Inpainted — absolutely overlaid, clipped to the slider's left side */}
          <div
            className="compare-reveal"
            style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
          >
            <img
              src={inpaintedUrl}
              alt="Inpainted"
              className="compare-img compare-img-overlay"
              draggable={false}
            />
          </div>

          {/* Vertical divider + circular drag handle */}
          <div className="compare-divider" style={{ left: `${sliderPos}%` }}>
            <div className="compare-handle">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                <path
                  d="M9 4L4 12L9 20M15 4L20 12L15 20"
                  stroke="white"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>
          </div>

          {/* Corner labels */}
          <span className="compare-label compare-label-left">Original</span>
          <span className="compare-label compare-label-right">Inpainted</span>
        </div>
      </div>
    </div>
  );
}
