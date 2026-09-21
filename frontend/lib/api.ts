/**
 * The API client, and the types the rest of the app renders.
 *
 * Every request goes to NEXT_PUBLIC_API_URL from the *browser*, not from the
 * Next server. That matters in compose: the containers can reach each other by
 * service name, but the code doing the fetching runs on your machine, so the
 * URL has to be one you can reach too.
 */

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export interface Job {
  id: string;
  url: string;
  video_id: string;
  status: JobStatus;
  stage: string | null;
  stage_label: string | null;
  stages_done: string[];
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  exit_code: number | null;
  /** Fraction of stages finished. Stage count, not elapsed time. */
  progress: number;
}

export interface Stage {
  id: string;
  label: string;
}

export interface Meta {
  stages: Stage[];
  emotions: string[];
  display_names: Record<string, string>;
  /** Canonical class to hex, read from the same table the PNG uses. */
  palette: Record<string, string>;
}

export interface Health {
  ok: boolean;
  downloads: string;
  can_run_pipeline: boolean;
  missing_checkpoints: string[];
}

export interface RunSummary {
  video_id: string;
  segments: number;
  has_timeline: boolean;
  duration_seconds: number;
  neutral_share: number;
  agreement: Record<string, number>;
  emotions: Record<string, number>;
}

export interface RunDetail extends RunSummary {
  per_model: Record<string, Record<string, number>>;
}

export interface Segment {
  segment_id: number | string;
  start_time: number;
  end_time: number;
  duration: number | null;
  text_ru: string;
  text_en: string;
  ru_valence: number | null;
  ru_arousal: number | null;
  en_valence: number | null;
  en_arousal: number | null;
  agreement: string;
  emotion_ru: string;
  emotion_ru_raw: string;
  emotion_en_distil: string;
  emotion_en_distil_raw: string;
  emotion_en_deberta: string;
  emotion_en_deberta_raw: string;
  emotion_final: string;
  emotion_final_raw: string;
  confidence: number | null;
}

/** A failed request carries the API's own reason, which is usually specific. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
      cache: "no-store",
    });
  } catch {
    // A network-level failure here almost always means the API is not running,
    // so say that rather than "Failed to fetch".
    throw new ApiError(`Cannot reach the API at ${API_URL}. Is it running?`, 0);
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      /* a non-JSON error body is not worth failing over */
    }
    throw new ApiError(detail, response.status);
  }
  return response.json() as Promise<T>;
}

export const api = {
  meta: () => request<Meta>("/api/meta"),
  health: () => request<Health>("/api/health"),

  listJobs: () => request<Job[]>("/api/jobs"),
  getJob: (id: string) => request<Job>(`/api/jobs/${id}`),
  submit: (url: string) =>
    request<Job>("/api/jobs", { method: "POST", body: JSON.stringify({ url }) }),
  cancel: (id: string) => request<{ cancelled: boolean }>(`/api/jobs/${id}`, { method: "DELETE" }),

  listRuns: () => request<RunSummary[]>("/api/runs"),
  getRun: (videoId: string) => request<RunDetail>(`/api/runs/${videoId}`),
  segments: (videoId: string) => request<Segment[]>(`/api/runs/${videoId}/segments`),
  timelineUrl: (videoId: string) => `${API_URL}/api/runs/${videoId}/timeline`,
  eventsUrl: (jobId: string) => `${API_URL}/api/jobs/${jobId}/events`,
};

/** `615.4` -> `10:15`. Used for segment boundaries, which are seconds. */
export function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "--:--";
  const total = Math.floor(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return hours > 0 ? `${hours}:${pad(minutes)}:${pad(secs)}` : `${minutes}:${pad(secs)}`;
}

/** Fallback colours, used only when the API has no matplotlib to read from. */
export const FALLBACK_PALETTE: Record<string, string> = {
  neutral: "#95a5a6",
  joy: "#FFD700",
  fear: "#9b59b6",
  anger: "#DC143C",
  surprise: "#9400D3",
  sadness: "#00008B",
  disgust: "#6B8E23",
};
