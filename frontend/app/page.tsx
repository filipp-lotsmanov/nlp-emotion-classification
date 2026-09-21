"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ApiError, api, type Health, type Job } from "@/lib/api";
import { StatusPill } from "@/components/Status";

/**
 * Submit a URL, and show what is already running.
 *
 * The form is disabled when the API reports a missing checkpoint. Letting
 * someone submit a run that cannot reach stage 6 and only finding out an hour
 * later is the kind of silent failure this project has spent its life removing.
 */
export default function SubmitPage() {
  const router = useRouter();
  const [url, setUrl] = useState("");
  const [health, setHealth] = useState<Health | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.health().then(setHealth).catch((e: ApiError) => setError(e.message));
    const load = () => api.listJobs().then(setJobs).catch(() => {});
    load();
    // Polled rather than streamed: this list is a summary, and one request
    // every five seconds is cheaper than twelve open SSE connections.
    const timer = setInterval(load, 5000);
    return () => clearInterval(timer);
  }, []);

  const active = jobs.filter((j) => j.status === "running" || j.status === "queued");

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const job = await api.submit(url.trim());
      router.push(`/jobs/${job.id}`);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
      setBusy(false);
    }
  }

  return (
    <>
      <h2>Analyse a video</h2>

      {error && <div className="notice error" style={{ marginBottom: 12 }}>{error}</div>}

      {health && !health.can_run_pipeline && (
        <div className="notice" style={{ marginBottom: 12 }}>
          <strong>The pipeline cannot run.</strong> Missing checkpoints:{" "}
          <span className="mono">{health.missing_checkpoints.join(", ")}</span>. Fetch them with{" "}
          <span className="mono">scripts/fetch_va_checkpoint.sh</span>, or browse{" "}
          <Link href="/runs">existing runs</Link> instead.
        </div>
      )}

      <form onSubmit={submit} className="card">
        <div className="row">
          <input
            type="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://www.youtube.com/watch?v=..."
            required
            style={{ flex: 1 }}
            disabled={busy || health?.can_run_pipeline === false}
          />
          <button type="submit" disabled={busy || !url.trim() || health?.can_run_pipeline === false}>
            {busy ? "Submitting…" : "Run pipeline"}
          </button>
        </div>
        <p className="hint">
          Russian-language video. Nine stages: download, scenes, audio, transcription, alignment,
          translation, valence-arousal, emotion, timeline. A two-hour recording takes roughly an
          hour on a GPU and most of a day without one. One job runs at a time — the rest queue.
        </p>
      </form>

      {active.length > 0 && (
        <>
          <h2>In progress</h2>
          <div className="grid">
            {active.map((job, index) => (
              <Link
                key={job.id}
                href={`/jobs/${job.id}`}
                className="card enter"
                style={{ textDecoration: "none", color: "inherit", "--i": index } as React.CSSProperties}
              >
                <div className="row" style={{ justifyContent: "space-between" }}>
                  <span className="mono">{job.video_id}</span>
                  <StatusPill status={job.status} />
                </div>
                <div
                  className={`bar${job.status === "running" ? " live" : ""}`}
                  style={{ marginTop: 10 }}
                >
                  <span style={{ width: `${Math.round(job.progress * 100)}%` }} />
                </div>
                <p className="hint">{job.stage_label ?? "queued"}</p>
              </Link>
            ))}
          </div>
        </>
      )}

      <h2>What this is</h2>
      <div className="card">
        <p style={{ marginTop: 0 }}>
          Every label here comes from one shared vocabulary, so the timeline image, the CSV and
          this page cannot disagree about what a class is called. That was not always true: the
          Russian model emits <span className="mono">enthusiasm</span>, which the pipeline used to
          rewrite to Neutral — 14.5% of a video, silently relabelled.
        </p>
        <p className="hint">
          Results skew heavily Neutral. That is the corpus, not a bug: it is 97% Twitter text and
          2.6% television dialogue, applied to documentary speech. See{" "}
          <span className="mono">docs/PROVENANCE.md</span>.
        </p>
      </div>
    </>
  );
}
