"use client";

import { useEffect, useRef } from "react";
import { animate, useInView } from "motion/react";

/**
 * A number that counts up to its value when it scrolls into view.
 *
 * Only for the headline figures. Counting up a table cell would be noise;
 * counting up "288 full agreement" gives the eye something to follow to a
 * number that is the point of the panel.
 *
 * Writes through a ref rather than through state: a 60fps re-render of the
 * surrounding component for a number that is decoration is not a trade worth
 * making.
 */
export function Counter({ value, duration = 0.9 }: { value: number; duration?: number }) {
  const ref = useRef<HTMLSpanElement>(null);
  const seen = useInView(ref, { once: true, margin: "-40px" });

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    if (!seen) {
      node.textContent = "0";
      return;
    }
    // Respecting the same preference the stylesheet does; motion honours it
    // for its own animations but this one writes text directly.
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      node.textContent = String(value);
      return;
    }
    const controls = animate(0, value, {
      duration,
      ease: [0.32, 0.72, 0.28, 1],
      onUpdate: (latest) => {
        node.textContent = String(Math.round(latest));
      },
    });
    return () => controls.stop();
  }, [seen, value, duration]);

  return <span ref={ref}>{value}</span>;
}
