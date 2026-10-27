// Export files for the importer and CLI tests: a minimal zip writer, and ChatGPT / Claude conversations.json content.
import { deflateRawSync } from 'node:zlib'

/** A zip archive of `files` ({name: Buffer}); deflated unless `stored`. */
export function makeZip(files, stored = false) {
  const locals = [], centrals = []
  let offset = 0
  for (const [name, data] of Object.entries(files)) {
    const body = stored ? data : deflateRawSync(data)
    const n = Buffer.from(name)
    const local = Buffer.alloc(30)
    local.writeUInt32LE(0x04034b50, 0); local.writeUInt16LE(20, 4); local.writeUInt16LE(stored ? 0 : 8, 8)
    local.writeUInt32LE(body.length, 18); local.writeUInt32LE(data.length, 22); local.writeUInt16LE(n.length, 26)
    const central = Buffer.alloc(46)
    central.writeUInt32LE(0x02014b50, 0); central.writeUInt16LE(20, 4); central.writeUInt16LE(20, 6)
    central.writeUInt16LE(stored ? 0 : 8, 10); central.writeUInt32LE(body.length, 20); central.writeUInt32LE(data.length, 24)
    central.writeUInt16LE(n.length, 28); central.writeUInt32LE(offset, 42)
    locals.push(local, n, body)
    centrals.push(central, n)
    offset += 30 + n.length + body.length
  }
  const cd = Buffer.concat(centrals)
  const end = Buffer.alloc(22)
  end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(Object.keys(files).length, 8); end.writeUInt16LE(Object.keys(files).length, 10)
  end.writeUInt32LE(cd.length, 12); end.writeUInt32LE(offset, 16)
  return Buffer.concat([...locals, cd, end])
}

const msg = (id, role, text, t, extra = {}) => ({ id, author: { role }, create_time: t,
  content: { content_type: 'text', parts: [text] }, recipient: 'all', ...extra })

/** A ChatGPT conversation: an edited first reply (two branches), a tool call and its output, a hidden context message. */
export function chatgptExport() {
  const mapping = {
    root: { id: 'root', message: null, parent: null, children: ['sys'] },
    sys: { id: 'sys', message: { ...msg('sys', 'system', 'You are ChatGPT.', 1.7e9), metadata: { is_visually_hidden_from_conversation: true } }, parent: 'root', children: ['ctx'] },
    ctx: { id: 'ctx', message: { ...msg('ctx', 'user', 'User profile: likes cats', 1.7e9), metadata: { is_visually_hidden_from_conversation: true } }, parent: 'sys', children: ['u1'] },
    u1: { id: 'u1', message: msg('u1', 'user', 'I adopted a cat named Mochi.', 1700000000), parent: 'ctx', children: ['a1old', 'a1'] },
    a1old: { id: 'a1old', message: msg('a1old', 'assistant', 'An old reply that was regenerated.', 1700000005), parent: 'u1', children: [] },
    a1: { id: 'a1', message: msg('a1', 'assistant', 'Mochi is a lovely name!', 1700000010), parent: 'u1', children: ['u2'] },
    u2: { id: 'u2', message: { ...msg('u2', 'user', '', 1700000100), content: { content_type: 'multimodal_text', parts: [{ asset_pointer: 'file-1' }, 'What does she eat? Plot it.'] } }, parent: 'a1', children: ['call'] },
    call: { id: 'call', message: { ...msg('call', 'assistant', 'plot()', 1700000110), recipient: 'python', content: { content_type: 'code', text: 'plot()' } }, parent: 'u2', children: ['tool'] },
    tool: { id: 'tool', message: msg('tool', 'tool', 'image', 1700000111), parent: 'call', children: ['a2'] },
    a2: { id: 'a2', message: msg('a2', 'assistant', 'Cats like Mochi eat tuna.', 1700000120), parent: 'tool', children: [] },
  }
  return [{ title: 'Cat', create_time: 1700000000, conversation_id: 'conv-1', current_node: 'a2', mapping },
    { title: 'Empty', create_time: 1700000000, conversation_id: 'conv-2', current_node: 'r', mapping: { r: { id: 'r', message: null, parent: null } } }]
}

export function claudeExport() {
  return [{ uuid: 'c-1', name: 'Trip', created_at: '2026-05-01T09:00:00Z', chat_messages: [
    { uuid: 'm1', sender: 'human', text: 'We are going to Hakodate in June.', created_at: '2026-05-01T09:00:00Z' },
    { uuid: 'm2', sender: 'assistant', text: '', content: [{ type: 'text', text: 'Hakodate in June is nice.' }, { type: 'tool_use' }], created_at: '2026-05-01T09:00:05Z' },
    { uuid: 'm3', sender: 'human', text: '   ', created_at: '2026-05-01T09:01:00Z' },
  ] }]
}
