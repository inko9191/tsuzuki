#!/usr/bin/env node
// `tsuzuki-memory` command: the memory sidecar (lib/server.js). A small file that exists before the build, so
// `pnpm install` of a fresh checkout can link the command; it explains what is missing instead of failing obscurely.
import { existsSync } from 'node:fs'

const server = new URL('../lib/server.js', import.meta.url)
if (!existsSync(server)) {
  process.stderr.write('tsuzuki-memory: the sidecar is not built yet (lib/server.js is missing). Run '
    + '"npx -y pnpm@10.26.1 run build" in dsh-plugins/, or install the packed dsh-tsuzuki tarball.\n')
  process.exit(1)
}
const { main } = await import(server.href)
await main()
