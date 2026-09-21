/** Reading a pipeline log well enough to tell someone what broke. */

/**
 * The `<time> - LEVEL - logger - ` prefix the pipeline's own lines carry.
 * yt-dlp's output and Python tracebacks have no prefix, and their text is
 * kept whole.
 */
const PREFIX = /^\S+ \S+ - (\w+) - [\w.]+ -\s*/;

/** The orchestrator's trailer, and traceback scaffolding. Never the reason. */
const NOISE = /^(Pipeline stopped|Traceback \(most recent)|^\s*File "|^\s*\^+\s*$|^\s*at /;

/** An unprefixed line that is itself an error: yt-dlp's, or Python's. */
const BARE_ERROR = /^(ERROR:|[A-Za-z]*(Error|Exception)\b|Refusing to start)/;

const MAX = 400;

/**
 * The line of a failed run's log that actually says what went wrong.
 *
 * `job.error` is "pipeline exited with code 1", which tells a reader nothing.
 * The reason is always in the log — "Sign in to confirm your age", "Refusing
 * to start: checkpoints ... are missing", "libcudnn.so.9: cannot open shared
 * object file" — and putting it in front of them is the difference between
 * the log being required reading and being optional.
 *
 * Scanned backwards, because the last error is the one that stopped the run.
 * Two passes: a real ERROR line first, then anything error-shaped. Without
 * the split, a run whose log ends "Failed: 1" reports that instead of the
 * reason three lines above it.
 */
export function failureReason(lines: string[]): string | null {
  return scan(lines, true) ?? scan(lines, false);
}

function scan(lines: string[], strict: boolean): string | null {
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i].trim();
    if (!line || NOISE.test(line)) continue;

    const match = PREFIX.exec(line);
    const level = match?.[1];
    const message = (match ? line.slice(match[0].length) : line).trim();
    // A bare "… - ERROR - vea.pipeline -" with nothing after it is a spacer
    // the logger emits before a multi-line message.
    if (!message) continue;

    const isError =
      level === "ERROR" || level === "CRITICAL" || (!level && BARE_ERROR.test(message));
    if (strict ? !isError : !(isError || /error|failed|exception/i.test(message))) continue;

    return message.length > MAX ? `${message.slice(0, MAX)}…` : message;
  }
  return null;
}
