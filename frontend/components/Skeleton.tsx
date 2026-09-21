/**
 * Placeholder blocks, shown while a fetch is in flight.
 *
 * The pages used to render the word "Loading…" and then swap the whole layout
 * in at once, which reads as a flash. A block the size of the thing that is
 * coming keeps the page still.
 */
export function Skeleton({ height = 16, width = "100%" }: { height?: number; width?: string }) {
  return <div className="skeleton" style={{ height, width }} aria-hidden />;
}

/** A card-shaped group, for a list that has not arrived yet. */
export function SkeletonRows({ rows = 4 }: { rows?: number }) {
  // Uneven widths: a stack of identical bars looks like a rendering fault
  // rather than like text that has not loaded.
  const widths = ["78%", "92%", "64%", "85%", "71%", "88%"];
  return (
    <div role="status" aria-label="Loading">
      {Array.from({ length: rows }, (_, index) => (
        <Skeleton key={index} width={widths[index % widths.length]} />
      ))}
    </div>
  );
}
