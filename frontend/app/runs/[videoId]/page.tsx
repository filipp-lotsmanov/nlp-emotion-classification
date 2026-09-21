"use client";

import { use, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  ApiError,
  api,
  formatTime,
  FALLBACK_PALETTE,
  type Meta,
  type RunDetail,
  type Segment,
} from "@/lib/api";
import { Skeleton, SkeletonRows } from "@/components/Skeleton";
import { Timeline } from "@/components/Timeline";
import { Counter } from "@/components/Counter";

const PAGE = 200;

/**
 * `emotion_ru` -> the headings the tables use. Mirrors runs.MODEL_COLUMNS.
 *
 * `short` is the checkpoint, not the language: two of the three are English,
 * and a column headed "English" twice tells you nothing about which model
 * disagreed with which.
 */
const MODELS: { key: keyof Segment; raw: keyof Segment; label: string; short: string }[] = [
  { key: "emotion_ru", raw: "emotion_ru_raw", label: "Russian (rubert-tiny2)", short: "rubert-tiny2" },
  {
    key: "emotion_en_distil",
    raw: "emotion_en_distil_raw",
    label: "English (DistilRoBERTa)",
    short: "DistilRoBERTa",
  },
  {
    key: "emotion_en_deberta",
    raw: "emotion_en_deberta_raw",
    label: "English (DeBERTa)",
    short: "DeBERTa",
  },
];

function Swatch({ color }: { color: string }) {
  return <span className="swatch" style={{ background: color }} />;
}

/**
 * One run's results.
 *
 * Three models, one vocabulary. Where a model's own output differs from the
 * canonical class - `enthusiasm` against Joy - the raw string is shown next to
 * it rather than hidden, because that gap is the finding, not a rendering
 * detail. See docs/PROVENANCE.md section 13.
 */
export default function RunPage({ params }: { params: Promise<{ videoId: string }> }) {
  const { videoId } = use(params);
  const [run, setRun] = useState<RunDetail | null>(null);
  const [segments, setSegments] = useState<Segment[] | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [emotion, setEmotion] = useState("");
  const [agreement, setAgreement] = useState("");
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(PAGE);
  // The distribution bars grow from zero on first paint. It is the one
  // place motion carries meaning here: the eye reads the relative lengths
  // while they move, which a static bar chart makes you do deliberately.
  const [grown, setGrown] = useState(false);
  // A segment picked on the timeline: the table scrolls to it and holds a
  // highlight, which is the whole point of the strip being clickable.
  const [selected, setSelected] = useState<number | string | null>(null);
  const [showFigure, setShowFigure] = useState(false);

  useEffect(() => {
    api.meta().then(setMeta).catch(() => {});
    api.getRun(videoId).then(setRun).catch((e: ApiError) => setError(e.message));
    api.segments(videoId).then(setSegments).catch((e: ApiError) => setError(e.message));
  }, [videoId]);

  const palette = meta?.palette ?? FALLBACK_PALETTE;
  const names = meta?.display_names ?? {};
  const name = (label: string) => names[label] ?? (label || "—");
  const colour = (label: string) => palette[label] ?? "var(--text-dim)";

  /**
   * Is a model's own label worth showing next to the canonical one?
   *
   * Only when it is a different *word*. `Happiness` against `joy` is this
   * project's display name for the same class and showing it reads as a
   * defect; `enthusiasm` against `joy` is the model using a class nobody else
   * has, which is the thing worth seeing.
   */
  const rawIsInteresting = (raw: string, label: string) => {
    if (!raw || !label) return false;
    const lower = raw.toLowerCase();
    return lower !== label && lower !== (names[label] ?? "").toLowerCase();
  };

  const filtered = useMemo(() => {
    if (!segments) return [];
    const needle = query.trim().toLowerCase();
    return segments.filter((s) => {
      if (emotion && s.emotion_final !== emotion) return false;
      if (agreement && s.agreement !== agreement) return false;
      if (needle && !`${s.text_ru} ${s.text_en}`.toLowerCase().includes(needle)) return false;
      return true;
    });
  }, [segments, emotion, agreement, query]);

  // Reset the window whenever the filter changes, or "show more" would carry
  // an old offset into a shorter list.
  useEffect(() => setLimit(PAGE), [emotion, agreement, query]);

  useEffect(() => {
    // Two frames: one for the zero-width paint, one for the transition to
    // have something to animate from.
    const frame = requestAnimationFrame(() => requestAnimationFrame(() => setGrown(true)));
    return () => cancelAnimationFrame(frame);
  }, []);

  function pick(segment: Segment) {
    // Clear any filter that would hide the row we are about to scroll to.
    if (emotion && segment.emotion_final !== emotion) setEmotion("");
    if (agreement && segment.agreement !== agreement) setAgreement("");
    setQuery("");
    setSelected(segment.segment_id);
    // Next frame: the row may not exist yet if a filter was just cleared.
    requestAnimationFrame(() => {
      document
        .getElementById(`segment-${segment.segment_id}`)
        ?.scrollIntoView({ block: "center", behavior: "smooth" });
    });
  }

  if (error) return <div className="notice error">{error}</div>;
  if (!run)
    return (
      <div style={{ paddingTop: 8 }}>
        <Skeleton height={22} width="220px" />
        <Skeleton height={150} />
        <SkeletonRows rows={5} />
      </div>
    );

  const total = Object.values(run.emotions).reduce((a, b) => a + b, 0);
  const ranked = Object.entries(run.emotions).sort((a, b) => b[1] - a[1]);
  const agreements = Object.entries(run.agreement).sort((a, b) => b[1] - a[1]);
  const classes = meta?.emotions ?? ranked.map(([label]) => label);

  return (
    <>
      <h2 style={{ marginTop: 0 }}>
        <span className="mono">{run.video_id}</span>
      </h2>
      <p className="hint" style={{ marginBottom: 16 }}>
        {run.segments} segments over {formatTime(run.duration_seconds)} ·{" "}
        {(run.neutral_share * 100).toFixed(1)}% Neutral ·{" "}
        <a href={`https://www.youtube.com/watch?v=${run.video_id}`} target="_blank" rel="noreferrer">
          source video
        </a>{" "}
        · <Link href="/runs">all runs</Link>
      </p>

      <div className="row" style={{ justifyContent: "space-between", marginTop: 26 }}>
        <h3 style={{ margin: 0 }}>Timeline</h3>
        <div className="row" style={{ gap: 6 }}>
          <button
            className={`chip${showFigure ? "" : " on"}`}
            onClick={() => setShowFigure(false)}
          >
            Interactive
          </button>
          <button
            className={`chip${showFigure ? " on" : ""}`}
            onClick={() => setShowFigure(true)}
            disabled={!run.has_timeline}
            title={run.has_timeline ? "The PNG stage 8 wrote" : "Stage 8 produced no image"}
          >
            Report figure
          </button>
        </div>
      </div>

      {showFigure ? (
        <div className="figure">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={api.timelineUrl(run.video_id)} alt={`Emotion timeline for ${run.video_id}`} />
        </div>
      ) : segments ? (
        <>
          <Timeline
            segments={segments}
            palette={palette}
            displayName={name}
            focus={emotion || null}
            selected={selected}
            onSelect={pick}
          />
          <div className="legend">
            {ranked.map(([label, count]) => (
              <button
                key={label}
                className={`chip${emotion === label ? " on" : ""}`}
                onClick={() => setEmotion(emotion === label ? "" : label)}
                title={`Show only ${name(label)}`}
              >
                <span className="swatch" style={{ background: colour(label) }} />
                {name(label)}
                <span className="muted mono" style={{ marginLeft: 2 }}>
                  {count}
                </span>
              </button>
            ))}
          </div>
        </>
      ) : (
        <Skeleton height={100} />
      )}
      <p className="hint">
        {showFigure
          ? "Stage 8's own PNG, the same bytes that go in the report."
          : "Hover a band for the line that produced it; click to open its row. The colours are stage 8's."}
      </p>

      <h3>Final label distribution</h3>
      <div className="card">
        {ranked.map(([label, count], index) => (
          <div
            key={label}
            className="enter"
            style={{ marginBottom: 10, "--i": index } as React.CSSProperties}
          >
            <div className="row" style={{ justifyContent: "space-between", gap: 8 }}>
              <span>
                <Swatch color={colour(label)} />
                {name(label)}
              </span>
              <span className="muted mono">
                {count} · {total ? ((count / total) * 100).toFixed(1) : "0.0"}%
              </span>
            </div>
            <div className="bar" style={{ marginTop: 3 }}>
              <span
                style={{
                  width: grown && total ? `${(count / total) * 100}%` : "0%",
                  background: colour(label),
                  transitionDelay: `${Math.min(index, 8) * 45}ms`,
                }}
              />
            </div>
          </div>
        ))}
      </div>

      <h3>Per model</h3>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            {classes.map((label) => (
              <th key={label} className="num" title={name(label)}>
                <Swatch color={colour(label)} />
                {name(label)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Object.entries(run.per_model).map(([model, counts], index) => (
            <tr key={model} className="enter" style={{ "--i": index } as React.CSSProperties}>
              <td>{model}</td>
              {classes.map((label) => (
                <td key={label} className="num">
                  {counts[label] ? counts[label] : <span className="muted">0</span>}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="hint">
        Counts are canonical classes, so a model that emits{" "}
        <span className="mono">enthusiasm</span> is counted under Happiness here and in the CSV
        alike.
      </p>

      <h3>Agreement</h3>
      <div className="grid cols-3">
        {agreements.map(([kind, count], index) => (
          <div key={kind} className="card enter" style={{ "--i": index } as React.CSSProperties}>
            <div style={{ fontSize: 26, fontWeight: 650, letterSpacing: "-0.02em" }}>
              <Counter value={count} />
            </div>
            <div className="muted" style={{ fontSize: 13 }}>
              {kind.replace(/_/g, " ")}
            </div>
          </div>
        ))}
      </div>
      <p className="hint">
        Full agreement means all three models chose the same class; majority means two of three;
        the rest fell back to the most confident single prediction.
      </p>

      <h3>Segments</h3>
      <div className="row" style={{ marginBottom: 10 }}>
        <select value={emotion} onChange={(e) => setEmotion(e.target.value)}>
          <option value="">every emotion</option>
          {classes.map((label) => (
            <option key={label} value={label}>
              {name(label)}
            </option>
          ))}
        </select>
        <select value={agreement} onChange={(e) => setAgreement(e.target.value)}>
          <option value="">every agreement</option>
          {agreements.map(([kind]) => (
            <option key={kind} value={kind}>
              {kind.replace(/_/g, " ")}
            </option>
          ))}
        </select>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="search the transcript"
          style={{ flex: 1 }}
        />
      </div>

      {!segments ? (
        <SkeletonRows rows={6} />
      ) : (
        <>
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Transcript</th>
                <th className="num">V / A</th>
                {MODELS.map((model) => (
                  <th key={String(model.key)} title={model.label}>
                    {model.short}
                  </th>
                ))}
                <th>Final</th>
              </tr>
            </thead>
            <tbody>
              {filtered.slice(0, limit).map((segment, index) => (
                <tr
                  key={segment.segment_id}
                  id={`segment-${segment.segment_id}`}
                  className={`enter${selected === segment.segment_id ? " picked" : ""}`}
                  style={{ "--i": index } as React.CSSProperties}
                >
                  <td className="mono" style={{ whiteSpace: "nowrap" }}>
                    <button
                      className="linklike"
                      onClick={() => setSelected(segment.segment_id)}
                      title="Highlight this segment"
                    >
                      {formatTime(segment.start_time)}
                    </button>
                  </td>
                  <td style={{ maxWidth: 380 }}>
                    <div>{segment.text_en}</div>
                    <div className="muted" style={{ fontSize: 12.5 }}>
                      {segment.text_ru}
                    </div>
                  </td>
                  <td className="num mono" style={{ whiteSpace: "nowrap" }}>
                    {segment.en_valence?.toFixed(2) ?? "—"} / {segment.en_arousal?.toFixed(2) ?? "—"}
                  </td>
                  {MODELS.map((model) => {
                    const label = String(segment[model.key] ?? "");
                    const raw = String(segment[model.raw] ?? "");
                    return (
                      <td key={String(model.key)} style={{ whiteSpace: "nowrap" }}>
                        <Swatch color={colour(label)} />
                        {name(label)}
                        {/* Shown only when the model used a different word,
                            which is the whole reason the raw value is carried
                            through the API. */}
                        {rawIsInteresting(raw, label) && (
                          <span className="muted mono" title="the model's own label">
                            {" "}
                            ({raw})
                          </span>
                        )}
                      </td>
                    );
                  })}
                  <td style={{ whiteSpace: "nowrap" }}>
                    <Swatch color={colour(segment.emotion_final)} />
                    {name(segment.emotion_final)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {filtered.length === 0 && <p className="muted">No segment matches that filter.</p>}
          {filtered.length > limit && (
            <p>
              <button className="secondary" onClick={() => setLimit(limit + PAGE)}>
                Show {Math.min(PAGE, filtered.length - limit)} more
              </button>{" "}
              <span className="hint">
                showing {limit} of {filtered.length}
              </span>
            </p>
          )}
        </>
      )}

      <p className="hint" style={{ marginTop: 16 }}>
        Valence and arousal are the English model's. The published checkpoint separates valence
        well and arousal barely — AUC 0.5734, close to chance — so treat the arousal column as
        indicative only. See <span className="mono">docs/PROVENANCE.md</span> section 9.
      </p>
    </>
  );
}
