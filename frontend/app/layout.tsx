import type { Metadata } from "next";
import Link from "next/link";
import { Logo } from "@/components/Logo";
import { ThemeToggle } from "@/components/ThemeToggle";
import "./globals.css";

export const metadata: Metadata = {
  title: "Video Emotion Analysis",
  description: "Nine-stage emotion analysis of Russian video: submit, watch, read.",
};

/**
 * Applies a stored theme before the first paint.
 *
 * It has to be inline and blocking. React runs after paint, so reading
 * localStorage in an effect would render one frame in the system theme and
 * then swap - the flash this exists to avoid. Wrapped in try/catch because
 * storage throws rather than returning null when a browser has it disabled.
 */
const THEME_SCRIPT = `try{var t=localStorage.getItem("vea-theme");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // suppressHydrationWarning: the script above mutates data-theme on this
    // element before React hydrates, which React would otherwise report as a
    // server/client mismatch.
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body>
        <header className="site">
          <div className="inner">
            <h1>
              <Logo />
              Video Emotion Analysis
            </h1>
            <nav>
              <Link href="/">Submit</Link>
              <Link href="/runs">Runs</Link>
              <Link href="/jobs">Jobs</Link>
            </nav>
            <ThemeToggle />
          </div>
        </header>
        <main className="shell">{children}</main>
      </body>
    </html>
  );
}
