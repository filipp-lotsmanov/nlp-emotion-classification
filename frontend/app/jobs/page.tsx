"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ApiError, api, type Job } from "@/lib/api";
import { StatusPill } from "@/components/Status";
import { SkeletonRows } from "@/components/Skeleton";

export default function JobsPage() {
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = () =>
      api
        .listJobs()
        .then(setJobs)
        .catch((e: ApiError) => setError(e.message));
    load();
    const timer = setInterval(load, 5000);
    return () => clearInterval(timer);
  }, []);

  if (error) return <div className="notice error">{error}</div>;
  if (!jobs)
    return (
      <>
        <h2>Jobs</h2>
        <SkeletonRows rows={3} />
      </>
    );

  return (
    <>
      <h2>Jobs</h2>
      {jobs.length === 0 ? (
        <div className="card">
          <p style={{ margin: 0 }}>
            Nothing submitted yet. <Link href="/">Start a run →</Link>
          </p>
        </div>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Video</th>
              <th>Status</th>
              <th>Stage</th>
              <th className="num">Progress</th>
              <th>Submitted</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job, index) => (
              <tr key={job.id} className="enter" style={{ "--i": index } as React.CSSProperties}>
                <td>
                  <Link href={`/jobs/${job.id}`} className="mono">
                    {job.video_id}
                  </Link>
                </td>
                <td>
                  <StatusPill status={job.status} />
                </td>
                <td className="muted">{job.stage_label ?? "—"}</td>
                <td className="num">{Math.round(job.progress * 100)}%</td>
                <td className="muted">{new Date(job.created_at).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="hint">
        Jobs survive an API restart and a closed browser: each one is a file under{" "}
        <span className="mono">/data/jobs/</span>. A job left running when the API stopped is
        marked failed rather than left to block the queue.
      </p>
    </>
  );
}
