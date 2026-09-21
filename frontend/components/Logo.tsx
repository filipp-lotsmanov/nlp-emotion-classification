/**
 * The mark: a speech waveform whose bars carry the emotion palette.
 *
 * It replaces a four-colour gradient square that read as a flag rather than
 * as a logo. This says what the tool does — it listens to speech and colours
 * it — and the hues are the pipeline's own, so the mark, the timeline and the
 * CSV agree even here.
 */
export function Logo({ size = 18 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
      style={{ flex: "none", display: "block" }}
    >
      <rect x="1.5" y="10" width="2.6" height="4" rx="1.3" fill="#95a5a6" />
      <rect x="5.7" y="6.5" width="2.6" height="11" rx="1.3" fill="#00008b" />
      <rect x="9.9" y="2.5" width="2.6" height="19" rx="1.3" fill="#ffd700" />
      <rect x="14.1" y="5" width="2.6" height="14" rx="1.3" fill="#dc143c" />
      <rect x="18.3" y="9" width="2.6" height="6" rx="1.3" fill="#9400d3" />
    </svg>
  );
}
