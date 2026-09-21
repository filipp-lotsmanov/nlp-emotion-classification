"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ApiError, api, formatTime, type RunSummary } from "@/lib/api";
import { SkeletonRows } from "@/components/Skeleton";

export default function RunsPage() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listRuns()
      .then(setRuns)
      .catch((e: ApiError) => setError(e.message));
  }, []);

  if (error) return <div className="notice error">{error}</div>;
  if (!runs)
    return (
      <>
        <h2>Finished runs</h2>
        <SkeletonRows rows={4} />
      </>
    );
  if (runs.length === 0) {
    return (
      <div className="card">
        <p style={{ margin: 0 }}>
          No finished runs. <Link href="/">Submit one →</Link>
        </p>
        <p className="hint">
          A run appears here once stage 9 has written its CSV.
        </p>
      </div>
    );
  }

  return (
    <>
      <h2>Finished runs</h2>
      <table>
        <thead>
          <tr>
            <th>Video</th>
            <th className="num">Segments</th>
            <th className="num">Duration</th>
            <th className="num">Neutral</th>
            <th>Timeline</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run, index) => (
            <tr key={run.video_id} className="enter" style={{ "--i": index } as React.CSSProperties}>
              <td>
                <Link href={`/runs/${run.video_id}`} className="mono">
                  {run.video_id}
                </Link>
              </td>
              <td className="num">{run.segments}</td>
              <td className="num">{formatTime(run.duration_seconds)}</td>
              <td className="num">{(run.neutral_share * 100).toFixed(1)}%</td>
              <td className="muted">{run.has_timeline ? "yes" : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="hint">
        A high Neutral share is expected. The English classifier's training corpus is 97% Twitter
        text and 2.6% television dialogue, and this pipeline analyses documentary speech.
      </p>
    </>
  );
}
