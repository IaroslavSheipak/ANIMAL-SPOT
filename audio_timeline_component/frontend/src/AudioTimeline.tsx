import React, { useEffect, useMemo, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import { Streamlit, ComponentProps } from "streamlit-component-lib";

type TimelineEvent = {
  id: string;
  begin_s: number;
  end_s: number;
  label: string;
  source: string;
  confidence?: number | null;
};

type Lane = {
  id: string;
  label: string;
  events: TimelineEvent[];
};

type AudioTimelineArgs = {
  audio_base64: string;
  audio_mime: string;
  duration_s: number;
  peaks: number[];
  lanes: Lane[];
  class_colors: Record<string, string>;
  show_ground_truth: boolean;
  show_predictions: boolean;
  selected_classes: string[];
  initial_viewport?: { start_s: number; end_s: number };
};

const DEFAULT_HEIGHT = 220;
const LANE_HEIGHT = 28;
const LANE_GAP = 10;
const LABEL_WIDTH = 140;
const MIN_PX_PER_SEC = 1;
const MAX_PX_PER_SEC = 400;
const ZOOM_SELECT_MIN_PX = 12;

const AudioTimeline: React.FC<ComponentProps> = ({ args }) => {
  const {
    audio_base64,
    audio_mime,
    duration_s,
    peaks,
    lanes,
    class_colors,
    initial_viewport,
  } = (args as AudioTimelineArgs) ?? {};

  const rootRef = useRef<HTMLDivElement | null>(null);
  const waveRef = useRef<HTMLDivElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const waveSurferRef = useRef<WaveSurfer | null>(null);
  const animationRef = useRef<number | null>(null);
  const mutationRef = useRef<MutationObserver | null>(null);
  const baselineZoomRef = useRef<number | null>(null);
  const zoomSelectRef = useRef({
    active: false,
    startX: 0,
    currentX: 0,
    justZoomed: false,
  });
  const dragRef = useRef({
    active: false,
    startX: 0,
    scrollLeft: 0,
    moved: false,
    justDragged: false,
  });

  const [isReady, setIsReady] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [viewport, setViewport] = useState({ start: 0, end: duration_s || 0 });
  const [hover, setHover] = useState<{
    x: number;
    y: number;
    event: TimelineEvent | null;
  } | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<TimelineEvent | null>(null);

  const audioUrl = useMemo(() => {
    if (!audio_base64) return "";
    return `data:${audio_mime || "audio/wav"};base64,${audio_base64}`;
  }, [audio_base64, audio_mime]);

  const visibleLanes = useMemo(() => {
    if (!lanes) return [] as Lane[];
    return lanes.filter((lane) => lane.events && lane.events.length > 0);
  }, [lanes]);

  const totalLaneHeight = useMemo(() => {
    return visibleLanes.length > 0
      ? visibleLanes.length * LANE_HEIGHT + (visibleLanes.length - 1) * LANE_GAP
      : LANE_HEIGHT;
  }, [visibleLanes.length]);

  const waveHeight = Math.max(DEFAULT_HEIGHT, totalLaneHeight + 120);

  const getPixelsPerSecond = () => {
    const wrapper = waveSurferRef.current?.getWrapper();
    if (!wrapper || !duration_s) return 1;
    const totalWidth = wrapper.scrollWidth || wrapper.clientWidth;
    return totalWidth / duration_s;
  };

  const applyZoom = (target: number, resetScroll: boolean) => {
    if (!waveSurferRef.current) return;
    const wrapper = waveSurferRef.current.getWrapper();
    waveSurferRef.current.zoom(target);
    if (resetScroll) {
      wrapper.scrollLeft = 0;
    }
    updateViewport();
    requestDraw();
  };

  const fitZoomToViewport = (setBaseline: boolean) => {
    if (!waveSurferRef.current || !duration_s) return;
    const wrapper = waveSurferRef.current.getWrapper();
    const target = Math.max(MIN_PX_PER_SEC, Math.min(MAX_PX_PER_SEC, wrapper.clientWidth / duration_s));
    if (setBaseline) {
      baselineZoomRef.current = target;
    }
    applyZoom(target, true);
    waveSurferRef.current.seekTo(0);
  };

  const updateViewport = () => {
    const wrapper = waveSurferRef.current?.getWrapper();
    if (!wrapper || !duration_s) return;
    const pxPerSec = getPixelsPerSecond();
    const start = wrapper.scrollLeft / pxPerSec;
    const end = (wrapper.scrollLeft + wrapper.clientWidth) / pxPerSec;
    setViewport({ start: Math.max(0, start), end: Math.min(duration_s, end) });
  };

  const ensureOverlayCanvas = () => {
    const wrapper = waveSurferRef.current?.getWrapper();
    if (!wrapper) return null;
    if (!overlayRef.current) {
      const canvas = document.createElement("canvas");
      canvas.className = "overlay-canvas";
      canvas.style.position = "absolute";
      canvas.style.top = "0";
      canvas.style.left = "0";
      canvas.style.pointerEvents = "none";
      wrapper.style.position = "relative";
      wrapper.appendChild(canvas);
      overlayRef.current = canvas;
    }
    return overlayRef.current;
  };

  const drawOverlay = () => {
    const canvas = ensureOverlayCanvas();
    const wrapper = waveSurferRef.current?.getWrapper();
    if (!canvas || !wrapper) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = wrapper.scrollWidth || wrapper.clientWidth;
    const laneAreaHeight = totalLaneHeight;
    wrapper.style.paddingTop = `${laneAreaHeight + 12}px`;
    const height = wrapper.clientHeight || laneAreaHeight + 140;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const pxPerSec = getPixelsPerSecond();
    const start = wrapper.scrollLeft / pxPerSec;
    const end = (wrapper.scrollLeft + wrapper.clientWidth) / pxPerSec;

    // Grid lines every 5 seconds
    ctx.strokeStyle = "rgba(120,120,120,0.2)";
    ctx.lineWidth = 1;
    for (let t = Math.floor(start / 5) * 5; t <= end + 5; t += 5) {
      const x = t * pxPerSec;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }

    // Lanes
    visibleLanes.forEach((lane, idx) => {
      const laneTop = idx * (LANE_HEIGHT + LANE_GAP) + 6;
      ctx.fillStyle = "rgba(15,118,110,0.08)";
      ctx.fillRect(0, laneTop, canvas.width, LANE_HEIGHT);

      lane.events.forEach((evt) => {
        if (evt.end_s < start || evt.begin_s > end) return;
        const x = evt.begin_s * pxPerSec;
        const w = Math.max(1, (evt.end_s - evt.begin_s) * pxPerSec);
        const color = class_colors?.[evt.label] || "#6b7280";
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.65;
        ctx.fillRect(x, laneTop + 2, w, LANE_HEIGHT - 4);
        ctx.globalAlpha = 1.0;

        if (selectedEvent && selectedEvent.id === evt.id) {
          ctx.strokeStyle = "#111827";
          ctx.lineWidth = 2;
          ctx.strokeRect(x, laneTop + 1, w, LANE_HEIGHT - 2);
        }
      });

    });

    // Playhead
    const playX = currentTime * pxPerSec;
    ctx.strokeStyle = "#0f766e";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(playX, 0);
    ctx.lineTo(playX, canvas.height);
    ctx.stroke();

    const zoomSelect = zoomSelectRef.current;
    if (zoomSelect.active) {
      const x1 = Math.min(zoomSelect.startX, zoomSelect.currentX);
      const x2 = Math.max(zoomSelect.startX, zoomSelect.currentX);
      ctx.fillStyle = "rgba(15,118,110,0.12)";
      ctx.fillRect(x1, 0, x2 - x1, canvas.height);
      ctx.strokeStyle = "rgba(15,118,110,0.7)";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x1 + 0.5, 0.5, x2 - x1 - 1, canvas.height - 1);
    }
  };

  const requestDraw = () => {
    if (animationRef.current) return;
    animationRef.current = window.requestAnimationFrame(() => {
      animationRef.current = null;
      drawOverlay();
    });
  };

  const enforceSingleRowWave = () => {
    const wrapper = waveSurferRef.current?.getWrapper();
    if (!wrapper) return;
    wrapper.style.whiteSpace = "nowrap";
    wrapper.style.overflowY = "hidden";
    wrapper.style.display = "block";
    wrapper.style.fontSize = "0";

    wrapper.querySelectorAll("canvas").forEach((node) => {
      const element = node as HTMLElement;
      if (element.classList.contains("overlay-canvas")) return;
      element.style.display = "inline-block";
      element.style.verticalAlign = "top";
    });
  };

  const handleHover = (evt: MouseEvent) => {
    const wrapper = waveSurferRef.current?.getWrapper();
    const canvas = ensureOverlayCanvas();
    if (!wrapper || !canvas) return;
    const rect = canvas.getBoundingClientRect();
    const pxPerSec = getPixelsPerSecond();
    const x = evt.clientX - rect.left;
    const y = evt.clientY - rect.top;

    const laneIndex = Math.floor(y / (LANE_HEIGHT + LANE_GAP));
    const lane = visibleLanes[laneIndex];
    if (!lane) {
      setHover(null);
      return;
    }

    const time = x / pxPerSec;
    const match = lane.events.find((event) => time >= event.begin_s && time <= event.end_s);
    if (match) {
      const rootRect = rootRef.current?.getBoundingClientRect();
      const tooltipX = rootRect ? evt.clientX - rootRect.left : evt.clientX;
      const tooltipY = rootRect ? evt.clientY - rootRect.top : evt.clientY;
      setHover({ x: tooltipX, y: tooltipY, event: match });
    } else {
      setHover(null);
    }
  };

  const handleSelectEvent = (evt: MouseEvent) => {
    if (dragRef.current.justDragged) {
      dragRef.current.justDragged = false;
      return;
    }
    if (zoomSelectRef.current.justZoomed) {
      zoomSelectRef.current.justZoomed = false;
      return;
    }
    const wrapper = waveSurferRef.current?.getWrapper();
    const canvas = ensureOverlayCanvas();
    if (!wrapper || !canvas || !duration_s) return;
    const rect = canvas.getBoundingClientRect();
    const pxPerSec = getPixelsPerSecond();
    const x = evt.clientX - rect.left;
    const y = evt.clientY - rect.top;

    const laneIndex = Math.floor(y / (LANE_HEIGHT + LANE_GAP));
    const lane = visibleLanes[laneIndex];
    const time = x / pxPerSec;
    const clampedTime = Math.max(0, Math.min(duration_s, time));

    if (lane) {
      const match = lane.events.find((event) => time >= event.begin_s && time <= event.end_s);
      if (match) {
        setSelectedEvent(match);
        waveSurferRef.current?.seekTo(match.begin_s / duration_s);
        setCurrentTime(match.begin_s);
        requestDraw();
        return;
      }
    }

    waveSurferRef.current?.seekTo(clampedTime / duration_s);
    setCurrentTime(clampedTime);
    setSelectedEvent(null);
    requestDraw();
  };

  useEffect(() => {
    if (!waveRef.current || !audioUrl || !duration_s) return;

    waveSurferRef.current?.destroy();
    waveSurferRef.current = null;
    overlayRef.current = null;
    baselineZoomRef.current = null;
    setIsReady(false);
    setCurrentTime(0);
    setSelectedEvent(null);
    setHover(null);

    const ws = WaveSurfer.create({
      container: waveRef.current,
      height: waveHeight,
      waveColor: "#d08cff",
      progressColor: "rgba(0,0,0,0)",
      cursorColor: "rgba(0,0,0,0)",
      cursorWidth: 0,
      normalize: true,
      interact: true,
      minPxPerSec: MIN_PX_PER_SEC,
      scrollParent: true,
    });

    waveSurferRef.current = ws;

    ws.load(audioUrl, peaks && peaks.length > 0 ? peaks : undefined, duration_s);

    ws.on("ready", () => {
      setIsReady(true);
      updateViewport();
      Streamlit.setFrameHeight();
      enforceSingleRowWave();
      fitZoomToViewport(true);
      if (mutationRef.current) {
        mutationRef.current.disconnect();
      }
      mutationRef.current = new MutationObserver(() => {
        enforceSingleRowWave();
      });
      mutationRef.current.observe(wrapper, { childList: true, subtree: true });
      ensureOverlayCanvas();
      requestDraw();
    });

    ws.on("audioprocess", (time: number) => {
      setCurrentTime(time);
    });

    ws.on("play", () => {
      setIsPlaying(true);
      enforceSingleRowWave();
    });
    ws.on("pause", () => {
      setIsPlaying(false);
    });
    ws.on("zoom", () => {
      updateViewport();
      enforceSingleRowWave();
      requestDraw();
    });

    const wrapper = ws.getWrapper();
    const dragState = dragRef.current;
    wrapper.style.cursor = "grab";
    const handleScroll = () => {
      updateViewport();
      requestDraw();
    };
    const handleMouseLeave = () => {
      dragState.active = false;
      zoomSelectRef.current.active = false;
      wrapper.style.cursor = "grab";
      setHover(null);
    };
    const handleMouseDown = (evt: MouseEvent) => {
      if (evt.button !== 0) return;
      const rect = wrapper.getBoundingClientRect();
      const x = evt.clientX - rect.left + wrapper.scrollLeft;
      if (evt.shiftKey) {
        zoomSelectRef.current.active = true;
        zoomSelectRef.current.startX = x;
        zoomSelectRef.current.currentX = x;
        zoomSelectRef.current.justZoomed = false;
        requestDraw();
        return;
      }
      dragState.active = true;
      dragState.moved = false;
      dragState.justDragged = false;
      dragState.startX = evt.clientX;
      dragState.scrollLeft = wrapper.scrollLeft;
      wrapper.style.cursor = "grabbing";
    };
    const handleMouseMove = (evt: MouseEvent) => {
      if (zoomSelectRef.current.active) {
        const rect = wrapper.getBoundingClientRect();
        zoomSelectRef.current.currentX = evt.clientX - rect.left + wrapper.scrollLeft;
        requestDraw();
        return;
      }
      if (dragState.active) {
        const delta = evt.clientX - dragState.startX;
        if (Math.abs(delta) > 3) {
          dragState.moved = true;
        }
        wrapper.scrollLeft = dragState.scrollLeft - delta;
        updateViewport();
        requestDraw();
        return;
      }
      handleHover(evt);
    };
    const handleMouseUp = () => {
      if (zoomSelectRef.current.active && waveSurferRef.current) {
        const { startX, currentX } = zoomSelectRef.current;
        zoomSelectRef.current.active = false;
        const width = Math.abs(currentX - startX);
        if (width > ZOOM_SELECT_MIN_PX) {
          const pxPerSec = getPixelsPerSecond();
          const startTime = Math.min(startX, currentX) / pxPerSec;
          const endTime = Math.max(startX, currentX) / pxPerSec;
          const window = Math.max(0.5, endTime - startTime);
          const target = Math.max(MIN_PX_PER_SEC, Math.min(MAX_PX_PER_SEC, wrapper.clientWidth / window));
          waveSurferRef.current.zoom(target);
          wrapper.scrollLeft = Math.max(0, startTime * target);
          updateViewport();
          zoomSelectRef.current.justZoomed = true;
        } else {
          zoomSelectRef.current.justZoomed = false;
        }
        requestDraw();
      }
      if (!dragState.active) return;
      dragState.active = false;
      dragState.justDragged = dragState.moved;
      wrapper.style.cursor = "grab";
    };
    const handleWindowMouseUp = () => handleMouseUp();
    wrapper.addEventListener("scroll", handleScroll);
    wrapper.addEventListener("mousemove", handleMouseMove);
    wrapper.addEventListener("mouseleave", handleMouseLeave);
    wrapper.addEventListener("mousedown", handleMouseDown);
    wrapper.addEventListener("mouseup", handleMouseUp);
    wrapper.addEventListener("click", handleSelectEvent);
    window.addEventListener("mouseup", handleWindowMouseUp);

    return () => {
      wrapper.removeEventListener("scroll", handleScroll);
      wrapper.removeEventListener("mousemove", handleMouseMove);
      wrapper.removeEventListener("mouseleave", handleMouseLeave);
      wrapper.removeEventListener("mousedown", handleMouseDown);
      wrapper.removeEventListener("mouseup", handleMouseUp);
      wrapper.removeEventListener("click", handleSelectEvent);
      window.removeEventListener("mouseup", handleWindowMouseUp);
      ws.destroy();
      overlayRef.current = null;
      if (mutationRef.current) {
        mutationRef.current.disconnect();
        mutationRef.current = null;
      }
    };
  }, [audioUrl, duration_s, peaks, waveHeight]);

  useEffect(() => {
    if (!isReady) return;
    requestDraw();
  }, [currentTime, visibleLanes, class_colors, selectedEvent, isReady]);

  useEffect(() => {
    if (!isReady || !initial_viewport || !duration_s) return;
    const wrapper = waveSurferRef.current?.getWrapper();
    if (!wrapper) return;
    const pxPerSec = getPixelsPerSecond();
    const scrollLeft = initial_viewport.start_s * pxPerSec;
    wrapper.scrollLeft = scrollLeft;
    updateViewport();
    requestDraw();
  }, [isReady, initial_viewport, duration_s]);

  const togglePlay = () => {
    if (!waveSurferRef.current) return;
    waveSurferRef.current.playPause();
  };

  const resetZoom = () => {
    const baseline = baselineZoomRef.current;
    if (baseline !== null) {
      applyZoom(baseline, true);
      enforceSingleRowWave();
      return;
    }
    fitZoomToViewport(true);
    enforceSingleRowWave();
  };

  const handleWheel = (evt: React.WheelEvent) => {
    if (!waveSurferRef.current) return;
    evt.preventDefault();
    const delta = evt.deltaY;
    const wrapper = waveSurferRef.current.getWrapper();
    const pxPerSec = getPixelsPerSecond();
    const mouseX = evt.clientX - wrapper.getBoundingClientRect().left + wrapper.scrollLeft;
    const mouseTime = mouseX / pxPerSec;

    const nextPxPerSec = Math.max(MIN_PX_PER_SEC, Math.min(MAX_PX_PER_SEC, pxPerSec * (delta > 0 ? 0.9 : 1.1)));
    waveSurferRef.current.zoom(nextPxPerSec);

    const newScrollLeft = mouseTime * nextPxPerSec - (evt.clientX - wrapper.getBoundingClientRect().left);
    wrapper.scrollLeft = Math.max(0, newScrollLeft);
    updateViewport();
    requestDraw();
  };

  useEffect(() => {
    Streamlit.setFrameHeight();
  }, [visibleLanes.length, waveHeight]);

  return (
    <div className="timeline-root" ref={rootRef}>
      <div className="timeline-toolbar">
        <button className="btn" onClick={togglePlay} disabled={!isReady}>
          {isPlaying ? "Pause" : "Play"}
        </button>
        <button className="btn secondary" onClick={resetZoom} disabled={!isReady}>
          Reset Zoom
        </button>
        <div className="hint">Drag to pan · Shift+drag to zoom</div>
        <div className="readout">
          <span>Playhead: {currentTime.toFixed(3)} s</span>
          <span>
            Viewport: {viewport.start.toFixed(2)} – {viewport.end.toFixed(2)} s
          </span>
        </div>
      </div>

      <div className="timeline-layout">
        <div className="lane-labels" style={{ width: LABEL_WIDTH }}>
          {visibleLanes.map((lane) => (
            <div key={lane.id} className="lane-label">
              {lane.label}
            </div>
          ))}
        </div>

        <div className="timeline-scroll" onWheel={handleWheel}>
          <div ref={waveRef} className="wave-container" />
        </div>
      </div>

      {hover && hover.event ? (
        <div className="tooltip" style={{ left: hover.x + 8, top: hover.y + 8 }}>
          <strong>{hover.event.label}</strong>
          <div>{hover.event.source}</div>
          <div>
            {hover.event.begin_s.toFixed(3)} – {hover.event.end_s.toFixed(3)} s
          </div>
          {hover.event.confidence !== undefined && hover.event.confidence !== null ? (
            <div>Conf: {hover.event.confidence.toFixed(3)}</div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
};

export default AudioTimeline;
