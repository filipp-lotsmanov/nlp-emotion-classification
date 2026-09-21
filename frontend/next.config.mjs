/** @type {import('next').NextConfig} */
const nextConfig = {
  // Produces .next/standalone: a server plus only the traced dependencies,
  // which is what the runtime stage of the Dockerfile copies. Without it the
  // image would carry the whole of node_modules.
  output: "standalone",
  reactStrictMode: true,
  // The API is a separate origin in development (3000 -> 8000) and behind the
  // same host in compose. Both are handled by NEXT_PUBLIC_API_URL rather than
  // by a rewrite, so the browser's requests are visible in the network tab
  // instead of being proxied through Next.
};

export default nextConfig;
