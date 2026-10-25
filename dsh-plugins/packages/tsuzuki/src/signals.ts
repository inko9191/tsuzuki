/**
 * A signal that aborts as soon as any of `sources` does (with that source's reason); `release()` detaches it from them.
 *
 * Unlike `AbortSignal.any`, nothing stays attached to a long-lived source once the request is over. Combining each
 * request with Memory's own close signal through `AbortSignal.any` kept about 2 KB per request reachable for the life
 * of the Memory (Node 22), and kept a closed Memory reachable too.
 */
export function linkedSignal(sources: ReadonlyArray<AbortSignal | null | undefined>): { signal: AbortSignal, release: () => void } {
  const controller = new AbortController()
  const attached: Array<[AbortSignal, () => void]> = []
  const release = () => {
    for (const [source, listener] of attached) source.removeEventListener('abort', listener)
    attached.length = 0
  }
  for (const source of sources) {
    if (!source) continue
    if (source.aborted) {
      controller.abort(source.reason)
      break
    }
    const listener = () => {
      controller.abort(source.reason)
      release()
    }
    source.addEventListener('abort', listener, { once: true })
    attached.push([source, listener])
  }
  if (controller.signal.aborted) release()
  return { signal: controller.signal, release }
}
