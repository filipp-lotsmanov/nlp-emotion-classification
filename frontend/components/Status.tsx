"use client";

import type { Job, JobStatus, Stage } from "@/lib/api";

export function StatusPill({ status }: { status: JobStatus }) {
  return <span className={`pill status-${status}`}>{status}</span>;
}

/**
 * The twelve stages as a strip, with the current one highlighted.
 *
 * Progress is stage count rather than elapsed time, and the caption says so:
 * the stages are wildly uneven. Transcription and translation are most of the
 * wall clock, so "6 of 12" does not mean halfway in minutes, and a bar that
 * implied otherwise would be lying for most of a two-hour run.
 */
export function StageStrip({ job, stages }: { job: Job; stages: Stage[] }) {
  const done = new Set(job.stages_done);
  return (
    <div>
      <div className="stages">
        {stages.map((stage, index) => {
          const state = done.has(stage.id) ? "done" : job.stage === stage.id ? "active" : "";
          return (
            <div
              key={stage.id}
              className={`stage enter ${state}`}
              style={{ "--i": index } as React.CSSProperties}
              title={stage.label}
            >
              <div style={{ fontWeight: 600 }}>{stage.id}</div>
              <div style={{ fontSize: 10, opacity: 0.75 }}>{stage.label}</div>
            </div>
          );
        })}
      </div>
      {/* `live` puts a slow sheen on the fill. Transcription and translation
          can hold one stage for most of an hour, and a bar that is merely
          stationary is indistinguishable from one that has died. */}
      <div className={`bar${job.status === "running" ? " live" : ""}`} style={{ marginTop: 12 }}>
        <span style={{ width: `${Math.round(job.progress * 100)}%` }} />
      </div>
      <p className="hint">
        {job.stages_done.length} of {stages.length} stages complete
        {job.stage_label ? ` — currently ${job.stage_label}` : ""}. Stage count, not time:
        transcription and translation take most of the wall clock.
      </p>
    </div>
  );
}
