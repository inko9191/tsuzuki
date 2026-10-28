# Changelog

## 0.2.1 (2026-10-28)

**No memory for messages that need none**
- Automatic recall (the chat proxy, Hermes, the dsh plugin) adds nothing when no earlier message is alike: below a cosine of 0.31 with EmbeddingGemma 300m (`recall.minSimilarity`). In the pre-registered test M2 (136 new users, Gemma 4 E2B answering), replies that refuse ("it is not in my memory") fell from 8.3% to 1.8% and all harmful replies from 14.9% to 6.4%, at the same accuracy on questions that need memory (52.2% → 51.5%, −0.7 points [−2.2, 0.0]). See the README's Results.
- 0.31 was measured with EmbeddingGemma 300m only, so with another embedding model the gate is off unless you set `recall.minSimilarity`; the memory server says so when it starts. The current conversation never counts as "alike".
- Searches the model asks for are not gated: the MCP `recall` tool now uses `/v1/memory/search`.
- The memory server's log has one line per automatic recall: how many earlier messages were added and how alike the closest one was (never their text).

**Facts from chat apps sooner**
- Through the proxy, a chat counts as ended after 5 quiet minutes (`proxy.segmentIdleMinutes`) and its facts are made then; it waited 30, because chat apps never say that a chat ended.

**Fixes**
- `node scripts/setup.mjs` on a Node that is too old (Ubuntu 22.04 ships Node 12) says so and how to update, instead of stopping at a syntax error.
- The chat setup takes a server on port 8080 for a model server only when it lists models; Open WebUI also uses 8080, and the proxy would have sent chats to it.
- On Windows, setup says to open a new terminal after `winget install`.
- The DeepSeek Harness installer (`install-desktop.sh`) and the dsh end-to-end tests install the current package versions (they named 0.1.0).
- No `ExperimentalWarning: SQLite` in the memory server's log.

**README**
- English results with Gemma 4 E2B: "at least as well as the whole history" (the pre-registered verdict), not "better".
- The llama.cpp embedding port is 8291 everywhere (the manual install said 8081).
- Which chat apps work with the proxy (apps where you set the URL, such as Open WebUI; not LM Studio's or Ollama's own chat windows), where the API key is, room in the model server's context, git as a prerequisite, how to remove tsuzuki, and that MCP clients on a cloud model send what they recall to that model.

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
