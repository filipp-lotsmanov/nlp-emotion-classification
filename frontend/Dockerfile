# syntax=docker/dockerfile:1.9
#
# The Next.js frontend, built standalone so the runtime image carries the
# server and the traced dependencies rather than the whole of node_modules.
#
#   docker build -t vea:web ./frontend
#
# It talks to the API over the network, so it needs no Python and shares
# nothing with the pipeline images.

ARG NODE_VERSION=22

FROM node:${NODE_VERSION}-alpine AS deps
WORKDIR /app
# Only the manifests, so a source edit does not reinstall node_modules.
COPY package.json package-lock.json* ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --no-audit --no-fund 2>/dev/null || npm install --no-audit --no-fund

FROM node:${NODE_VERSION}-alpine AS build
WORKDIR /app

# NEXT_PUBLIC_* is inlined into the client bundle by `next build`; it is not
# read at runtime, because the code using it executes in the browser, where
# this container's environment does not exist. So it has to be a build
# argument. Setting it on the running container instead does nothing at all -
# the page loads and every request quietly goes to the default address.
ARG NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL} \
    NEXT_TELEMETRY_DISABLED=1

COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# Fail the build rather than ship a bundle pointing somewhere else: the value
# must actually be in the client chunks, which is the only proof that the
# inlining happened.
RUN grep -rq "$NEXT_PUBLIC_API_URL" .next/static/chunks/ \
 || (echo "NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL was not inlined" >&2; exit 1)

FROM node:${NODE_VERSION}-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000
RUN addgroup -g 10002 web && adduser -D -u 10002 -G web web

# `output: "standalone"` in next.config.mjs produces exactly this layout: a
# server plus only the dependencies the build actually traced.
COPY --from=build --chown=web:web /app/.next/standalone ./
COPY --from=build --chown=web:web /app/.next/static ./.next/static
COPY --from=build --chown=web:web /app/public ./public

USER web
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:3000/').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["node", "server.js"]
