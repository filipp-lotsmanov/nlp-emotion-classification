"use client";

import { useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { formatTime, type Segment } from "@/lib/api";

const VIEW_W = 1000;
const VIEW_H = 100;
const TICKS = 8;

interface Props {
  segments: Segment[];
  palette: Record<string, string>;
  displayName: (label: string) => string;
  /** Dim everything that is not this class. Null shows all of them lit. */
  focus: string | null;
  selected: number | string | null;
  onSelect: (segment: Segment) => void;
}

/**
 * The run as a strip you can point at.
 *
 * Stage 8's PNG says the same thing and cannot be questioned: you see a red
 * band and there is no way to ask which sentence it was. Here every band is
 * the segment, so hovering gives you the line that produced it and clicking
 * takes you to its row.
 *
 * Drawn from the same `/api/runs/{id}/segments` rows the table renders, in
 * the same colours the PNG uses, so the two cannot disagree about a video.
 */
export function Timeline({ segments, palette, displayName, focus, selected, onSelect }: Props) {
  const [hover, setHover] = useState<Segment | null>(null);
  const wrap = useRef<HTMLDivElement>(null);

  const duration = useMemo(
    () => segments.reduce((max, s) => Math.max(max, s.end_time), 0) || 1,
    [segments],
  );

  // Precomputed geometry: recalculating 311 rects on every hover is what
  // makes a chart feel sticky.
  const bars = useMemo(
    () =>
      segments.map((segment) => {
        const x = (segment.start_time / duration) * VIEW_W;
        const raw = ((segment.end_time - segment.start_time) / duration) * VIEW_W;
        return {
          segment,
          x,
          // A one-second segment in a 23-minute video is 0.7px wide and
          // effectively invisible, and worse, unhoverable. A floor keeps
          // every segment reachable at the cost of a little overlap.
          width: Math.max(raw, 1.4),
          colour: palette[segment.emotion_final] ?? "var(--text-dim)",
        };
      }),
    [segments, duration, palette],
  );

  const ticks = useMemo(
    () => Array.from({ length: TICKS + 1 }, (_, i) => (duration / TICKS) * i),
    [duration],
  );

  // Tooltip position, clamped so it never leaves the card.
  const tipLeft = hover ? Math.min(Math.max((hover.start_time / duration) * 100, 8), 92) : 50;

  return (
    <div className="timeline" ref={wrap}>
      <svg
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        preserveAspectRatio="none"
        className="timeline-svg"
        role="img"
        aria-label={`Emotion over ${formatTime(duration)}, ${segments.length} segments`}
        onMouseLeave={() => setHover(null)}
      >
        {bars.map(({ segment, x, width, colour }) => {
          const dim = focus !== null && segment.emotion_final !== focus;
          const active = hover?.segment_id === segment.segment_id;
          const chosen = selected === segment.segment_id;
          return (
            <rect
              key={segment.segment_id}
              x={x}
              y={0}
              width={width}
              height={VIEW_H}
              fill={colour}
              opacity={dim ? 0.14 : active || chosen ? 1 : 0.85}
              style={{ cursor: "pointer", transition: "opacity 140ms" }}
              onMouseEnter={() => setHover(segment)}
              onClick={() => onSelect(segment)}
            />
          );
        })}

        {hover && (
          <rect
            x={(hover.start_time / duration) * VIEW_W - 0.6}
            y={0}
            width={Math.max(((hover.end_time - hover.start_time) / duration) * VIEW_W, 1.4) + 1.2}
            height={VIEW_H}
            fill="none"
            stroke="var(--text)"
            strokeWidth={2}
            vectorEffect="non-scaling-stroke"
            pointerEvents="none"
          />
        )}
      </svg>

      <div className="timeline-axis" aria-hidden>
        {ticks.map((t, i) => (
          <span key={i}>{formatTime(t)}</span>
        ))}
      </div>

      <AnimatePresence>
        {hover && (
          <motion.div
            className="timeline-tip"
            style={{ left: `${tipLeft}%` }}
            initial={{ opacity: 0, y: 6, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 4, scale: 0.98 }}
            transition={{ duration: 0.14, ease: [0.32, 0.72, 0.28, 1] }}
          >
            <div className="row" style={{ gap: 7, marginBottom: 5 }}>
              <span
                className="swatch"
                style={{ background: palette[hover.emotion_final], margin: 0 }}
              />
              <strong style={{ fontSize: 12.5 }}>{displayName(hover.emotion_final)}</strong>
              <span className="mono muted" style={{ marginLeft: "auto" }}>
                {formatTime(hover.start_time)}
              </span>
            </div>
            <div style={{ fontSize: 12.5 }}>{hover.text_en || "(no translation)"}</div>
            {hover.text_ru && (
              <div className="muted" style={{ fontSize: 12 }}>
                {hover.text_ru}
              </div>
            )}
            <div className="hint" style={{ marginTop: 5 }}>
              {hover.agreement.replace(/_/g, " ")} · click to open the segment
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
