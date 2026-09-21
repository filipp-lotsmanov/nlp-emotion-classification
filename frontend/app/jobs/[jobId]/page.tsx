"use client";

import { use, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ApiError, api, type Job, type Stage } from "@/lib/api";
import { StageStrip, StatusPill } from "@/components/Status";
import { failureReason } from "@/lib/logs";

const MAX_LOG_LINES = 500;


/**
 * Watch one run.
 *
 * The state lives on the server, in a file, so this page holds nothing that
 * matters. Refresh it, close it, come back tomorrow — it reconnects to the
 * same job and the run never noticed.
 */
export default function JobPage({ params }: { params: Promise<{ jobId: string }> }) {
  const { jobId } = use(params);
  const [job, setJob] = useState<Job | null>(null);
  const [stages, setStages] = useState<Stage[]>([]);
  const [lines, setLines] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [follow, setFollow] = useState(true);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.meta().then((m) => setStages(m.stages)).catch(() => {});
    api.getJob(jobId).then(setJob).catch((e: ApiError) => setError(e.message));
  }, [jobId]);

  useEffect(() => {
    const source = new EventSource(api.eventsUrl(jobId));

    source.addEventListener("job", (event) => {
      setJob(JSON.parse((event as MessageEvent).data));
    });
    source.addEventListener("log", (event) => {
      const line = JSON.parse((event as MessageEvent).data) as string;
      // Bounded: a full run emits tens of thousands of lines, and keeping them
      // all would grow the DOM until the tab stalls.
      setLines((prev) => {
        const next = [...prev, line];
        return next.length > MAX_LOG_LINES ? next.slice(-MAX_LOG_LINES) : next;
      });
    });
    source.addEventListener("end", () => source.close());
    source.onerror = () => {
      // EventSource reconnects on its own; a finished job simply has nothing
      // more to send, so this is not surfaced as an error.
      source.close();
    };

    return () => source.close();
  }, [jobId]);

  useEffect(() => {
    if (follow && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [lines, follow]);

  async function cancel() {
    try {
      await api.cancel(jobId);
      setJob(await api.getJob(jobId));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  }

  if (error) return <div className="notice error">{error}</div>;
  if (!job) return <p className="muted">Loading…</p>;

  const finished = ["succeeded", "failed", "cancelled"].includes(job.status);
  const reason = job.status === "failed" ? failureReason(lines) : null;

  return (
    <>
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 4 }}>
        <h2 style={{ margin: 0 }}>
          <span className="mono">{job.video_id}</span> <StatusPill status={job.status} />
        </h2>
        {!finished && (
          <button className="secondary" onClick={cancel}>
            Cancel
          </button>
        )}
      </div>
      <p className="hint" style={{ marginBottom: 16 }}>
        <a href={job.url} target="_blank" rel="noreferrer">
          {job.url}
        </a>{" "}
        · submitted {new Date(job.created_at).toLocaleString()}
      </p>

      {job.error && (
        <div className="notice error" style={{ marginBottom: 12 }}>
          <strong>{reason ?? job.error}</strong>
          {reason && (
            <div className="hint" style={{ marginTop: 4 }}>
              {job.error}. The technical log below has the full output.
            </div>
          )}
        </div>
      )}

      {job.status === "succeeded" && (
        <div className="notice ok" style={{ marginBottom: 12 }}>
          Finished. <Link href={`/runs/${job.video_id}`}>Open the results →</Link>
        </div>
      )}

      <div className="card">{stages.length > 0 && <StageStrip job={job} stages={stages} />}</div>

      {/* Closed unless the run failed. Someone watching a healthy run wants
          the stage strip; someone looking at a failed one wants this open
          already, without having to find it. */}
      <details className="fold" style={{ marginTop: 16 }} open={job.status === "failed"}>
        <summary>
          Technical log
          <span className="spacer" />
          <span className="muted" style={{ fontSize: 12 }}>
            {lines.length ? `${lines.length} line${lines.length === 1 ? "" : "s"}` : "no output yet"}
          </span>
        </summary>
        <div className="fold-body">
          <label className="hint" style={{ display: "block", marginBottom: 8 }}>
            <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} />{" "}
            follow new output
          </label>
          <div className="log" ref={logRef}>
            {lines.length === 0 ? (
              <span className="muted">
                {finished ? "No output captured." : "Waiting for the pipeline to start…"}
              </span>
            ) : (
              lines.join("\n")
            )}
          </div>
          <p className="hint">
            Last {MAX_LOG_LINES} lines. The full log is kept server-side as{" "}
            <span className="mono">{job.id}.log</span> in the jobs directory —{" "}
            <span className="mono">/data/jobs/</span> in the container,{" "}
            <span className="mono">data/jobs/</span> otherwise.
          </p>
        </div>
      </details>
    </>
  );
}
