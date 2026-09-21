"use client";

import { useEffect, useState } from "react";

type Theme = "light" | "dark";
const KEY = "vea-theme";

/**
 * Switch between the two themes, remembering the choice per browser.
 *
 * The starting theme is the system preference; clicking here overrides it and
 * the override is what gets stored. `layout.tsx` applies a stored value in a
 * blocking inline script, before first paint, so returning to the page does
 * not flash the wrong theme for a frame.
 *
 * The icon renders only after mount. The server cannot know which theme the
 * browser resolved, and rendering a guess produces a hydration mismatch and a
 * visible swap on every load.
 */
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme | null>(null);

  useEffect(() => {
    const stored = document.documentElement.dataset.theme as Theme | undefined;
    if (stored === "light" || stored === "dark") {
      setTheme(stored);
      return;
    }
    setTheme(window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  }, []);

  function toggle() {
    const next: Theme = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.dataset.theme = next;
    try {
      window.localStorage.setItem(KEY, next);
    } catch {
      // Private browsing, or storage denied. The theme still applies for this
      // page; only remembering it across loads is lost, which is not worth an
      // error message.
    }
  }

  return (
    <button
      type="button"
      className="icon-button"
      onClick={toggle}
      aria-label={theme === "dark" ? "Switch to the light theme" : "Switch to the dark theme"}
      title={theme === "dark" ? "Light theme" : "Dark theme"}
    >
      {theme === null ? (
        <span style={{ width: 16, height: 16, display: "block" }} aria-hidden />
      ) : theme === "dark" ? (
        <SunIcon />
      ) : (
        <MoonIcon />
      )}
    </button>
  );
}

function SunIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
      <circle cx="12" cy="12" r="4.2" />
      <path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.2 5.2l1.4 1.4M17.4 17.4l1.4 1.4M18.8 5.2l-1.4 1.4M6.6 17.4l-1.4 1.4" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" aria-hidden>
      <path d="M20 14.2A8.2 8.2 0 0 1 9.8 4a8.2 8.2 0 1 0 10.2 10.2z" />
    </svg>
  );
}
