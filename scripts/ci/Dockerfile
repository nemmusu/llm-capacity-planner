# Local CI: lint, types, tests with the 100 % coverage gate and the Linux single executable, in a clean Linux
# with the oldest supported Node (22). Run by `npm run ci:docker`; no GPU and no inference engine needed.
FROM node:22-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
  && rm -rf /var/lib/apt/lists/*
WORKDIR /work
