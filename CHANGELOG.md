# Changelog

## 0.2.0 (2026-10-27)

**Use it from more places**
- Chat apps on your own model server: an OpenAI-compatible proxy (`--for chat`) in front of llama.cpp, Ollama or LM Studio gives every chat memory, with no tool calls.
- MCP clients (Claude Desktop, Claude Code, Cursor, LM Studio, Jan, Cline...): `lib/mcp.js` with the tools `recall` and `remember` (`forget` with `--forget`).
- Your ChatGPT or Claude history: `node scripts/memory.mjs import <export.zip>`.

**Easier to set up and to test**
- The setup asks with words (`chat`, `mcp`, `hermes`, `app`), not numbers, and shows first what quick (334 MB, about 5 minutes) and full (3.7 GB) cost.
- At the end it shows the memory working on your machine: a new chat answers from an earlier one, next to the same model without memory.
- Every error says what to do next. English or Japanese, from the system's language (`--lang`).
- `--report` prints a test report to paste into an issue (OS, CPU, GPU, llama.cpp and its devices, the time of each step), also when setup fails.

**Keep it running**
- `--autostart` starts the memory server at login (macOS, Linux, Windows); `--update` gets the newest version, rebuilds and restarts; `--version`.
- `node scripts/memory.mjs`: `status`, `recent`, `show`, `search`, `forget`, `export`, `import`.

**Security** (from a review before this release)
- Clients ask the memory server to prove it knows the token before they send it (`/health?nonce=`), so a program that took the port while the memory was down gets neither the token nor a conversation, and `--stop` never stops it.
- The proxy does not follow redirects from the model server, and bounds the reply it keeps.
- Imports bound the size of what they unpack, and the terminal commands remove control characters before printing.
- The test report removes the home folder in every spelling and the user name in any case.

**Fixes**
- The Hermes provider restarts a dead memory server at once, also in the first minute after the computer starts.
- A fact extraction cut at the output limit is not retried three times (it would be cut again).
- MCP: a memory folder reached through a symbolic link (macOS `/var`) is no longer taken for another memory; recall leaves nothing out (one MCP process serves many chats), and forget takes a date or all.
- Proxy: a chat is found by its earlier messages, not by its system prompt and first message, so two chats that open with the same words stay apart, and a system prompt with the time in it or a trimmed history stores nothing twice.
- README: the write-time cost is measured one conversation at a time (median 4.2 s with Gemma 4 E2B; it said 10–14 s).

## 0.1.0 (2026-10-25)

First public preview: memory for DeepSeek Harness and Hermes Agent, and an HTTP sidecar; one-command setup with llama.cpp; pre-registered results E1, J1, J1-Q, U2, S2, S3.
