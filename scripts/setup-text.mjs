// What scripts/setup.mjs says, in English and Japanese (chosen by --lang, else by the system's language).
// Both objects have the same keys; tests/setup-lib.test.mjs checks that.

const mb = n => `${Math.round(n / 1e6).toLocaleString()} MB`
const gb = n => `${(n / 1e9).toFixed(2)} GB`

export const TEXT = {
  en: {
    help: v => `tsuzuki ${v} setup. Usage: node scripts/setup.mjs [options]

  --for chat|mcp|hermes|app
                         where the memory is used (asked if missing):
                           chat    chat apps on your own model server (llama.cpp, Ollama, LM Studio...)
                           mcp     an MCP client (Claude Desktop, Claude Code, Cursor, LM Studio, Jan, Cline...)
                           hermes  Hermes Agent
                           app     your own program, over HTTP
  --quick                only the search model (334 MB, about 5 minutes); no facts (same as --no-facts)
  --upstream URL         with --for chat: your model server's OpenAI-compatible URL (default: the first that
                         answers of llama.cpp :8080, Ollama :11434, LM Studio :1234)
  --profile NAME         Hermes profile (default: "default", the Hermes home itself)
  --hermes-home DIR      Hermes home (default: $HERMES_HOME, else ~/.hermes; on Windows %LOCALAPPDATA%\\hermes)
  --models DIR           where the models go (default: ~/.tsuzuki/models)
  --llama-server PATH    llama.cpp's llama-server, if it is not on PATH
  --report               at the end, print a test report to paste into an issue (works when setup fails too)
  --lang en|ja           the language of this setup (default: the system's)
  --yes                  do not ask: accept the defaults and the model licenses
  --skip-build           do not rebuild (if it is built); --skip-check: do not run the final check

  --start                start the memory server in the background, and exit
  --stop                 stop the memory server and the llama-servers it started, and exit
  --autostart[=off]      start the memory server when you log in (=off: no longer), and exit
  --update               get the newest version (git pull), rebuild, restart the memory server, show what changed
  --version, --help`,
    banner: v => `tsuzuki ${v} setup  (Ctrl+C stops it at any time; running it again continues)`,
    unknownArg: a => `unknown option: ${a} (see --help)`,
    needsValue: k => `--${k} needs a value`,
    nodeOld: (v, how) => `Node ${v} is too old: tsuzuki needs Node 22 (22.19 or later) or 24+.\n  Install it with:  ${how}\n  then run this again.`,
    nodeHow: { darwin: 'brew install node   (or the installer from https://nodejs.org)', win32: 'winget install OpenJS.NodeJS.LTS   (then open a new terminal: winget updates PATH only for new ones)',
      linux: 'the installer or a package from https://nodejs.org (distribution packages are often too old)' },
    menu: `
Where will you use the memory? Type one word:
  chat     chat apps on your own model server (llama.cpp, Ollama, LM Studio...): memory in every chat
  mcp      an MCP client: Claude Desktop, Claude Code, Cursor, LM Studio, Jan, Cline...
  hermes   Hermes Agent
  app      your own program (HTTP API on 127.0.0.1)
>`,
    whatTarget: 'where to use the memory (--for chat, mcp, hermes or app)',
    menuNumber: '  Type the word (chat, mcp, hermes or app). Older versions numbered these, and the numbers changed.',
    menuUnknown: a => `  "${a}" is not one of them: type chat, mcp, hermes or app.`,
    forInvalid: v => `--for must be chat, mcp, hermes or app, not "${v}"`,
    needFor: 'With --yes, say where the memory is used: --for chat, mcp, hermes or app.',
    planTitle: 'How much to set up',
    planQuick: (size) => `  quick   search only: downloads ${mb(size)}, about 5 minutes. Memory works on the messages themselves.`,
    planFull: (size) => `  full    search and facts: downloads ${gb(size)} in all, about 15–30 minutes the first time (mostly the download).`
      + '\n          Facts keep recall good as conversations pile up; this is the tested setting.',
    planQ: 'Type quick or full [full]:',
    whatPlan: 'quick or full (--quick for quick)',
    planUnknown: a => `  "${a}" is neither: type quick or full.`,
    buildStep: 'Building the plugins and the memory server',
    installing: 'installing packages', building: 'building', packing: 'packing (for DeepSeek Harness)',
    cmdFailed: (cmd, cwd, out) => `${cmd} failed in ${cwd}:\n${out}\n  What to do: check the network connection and run setup again. If it fails the same way, run it with --report and send the result.`,
    llamaStep: 'Looking for llama.cpp (llama-server)',
    llamaMissing: (where, how) => `llama-server was not found ${where}. Install llama.cpp first:\n    ${how}\n  then run this again (or pass --llama-server <path to llama-server>).`,
    llamaWhere: g => g ? `at ${g}` : 'on PATH',
    llamaHow: { darwin: 'brew install llama.cpp', win32: 'winget install llama.cpp   (or a Windows build from https://github.com/ggml-org/llama.cpp/releases),\n    then open a new terminal (winget updates PATH only for new ones)',
      linux: 'a Linux build from https://github.com/ggml-org/llama.cpp/releases (or brew install llama.cpp)' },
    llamaBad: (bin, why, text) => `${bin} --help did not work as expected (${why}):\n${text}\n  What to do: install a recent llama.cpp (one with --embedding) and pass its llama-server with --llama-server.`,
    modelsStep: dir => `Models (in ${dir})`,
    already: f => `  ${f}: already downloaded`,
    checking: (f, size) => `  ${f}: ${mb(size)}, checking...`,
    license: (title, size, license) => `  ${title}: ${mb(size)}.\n  It is under ${license}.`,
    downloadQ: '  Download it?',
    whatLicense: 'whether to download EmbeddingGemma (Gemma Terms of Use)',
    noEmbedding: 'search needs this model, so setup stopped here. Run it again when you want to continue.',
    quickNote: '  Quick setup: no facts model. Run setup again without --quick to add it later.',
    httpFailed: (s, url) => `download failed: HTTP ${s} for ${url}\n  What to do: check that the address opens in a browser, then run setup again.`,
    downloadStopped: (f, why) => `downloading ${f} stopped (${why}).\n  What to do: check the network connection and run setup again: it continues where it stopped.`,
    badFile: (f, d) => `${f} is not the expected file (sha256 ${d}).\n  What to do: run setup again to download it again.`,
    diskFull: (dir, need, free) => `not enough free disk space in ${dir}: ${gb(need)} needed, ${gb(free)} free.\n  What to do: free some space, or put the models elsewhere with --models <folder>.`,
    hermesStep: 'Connecting to Hermes Agent',
    profileQ: '  Hermes profile ("default" is the Hermes home itself) [default]:',
    whatProfile: 'the Hermes profile (--profile)',
    noHermes: p => `${p} was not found.\n  What to do: start Hermes once with this profile, or pass --hermes-home <folder>. (For another app than Hermes, use --for chat, mcp or app.)`,
    quitHermes: '  Quit Hermes if it is running, then press Enter.',
    whatQuit: 'that Hermes is not running',
    installerStopped: out => `the Hermes installer stopped:\n${out}`,
    noPython: 'Python 3 was not found (the Hermes installer needs it).\n  What to do: install Python 3, or update Hermes (recent versions bring their own Python).',
    memorySettings: (p, port) => `  memory settings: ${p} (port ${port})`,
    searchLine: '  search: EmbeddingGemma, started on demand by the memory',
    factsOn: '  facts: Gemma 4 E2B, started on demand on this machine (even if Hermes itself uses a cloud model)',
    factsOff: '  facts: off (run setup again without --quick to add the facts model)',
    appStep: 'Preparing the memory server for your program', mcpStep: 'Preparing the memory for MCP clients',
    chatStep: 'Preparing the memory proxy for your chat app',
    settings: (p, port, data) => `  settings: ${p} (port ${port}; memory in ${data})`,
    upstreamStep: 'Your model server',
    found: (name, url) => `  found ${name} at ${url}`,
    upstreamQ: d => `  Your model server's OpenAI-compatible URL [${d}]:`,
    whatUpstream: "your model server's URL (--upstream)",
    notUrl: u => `not a URL: ${u}`,
    notLocal: u => `${u} is not on this machine: recalled memory would be sent there.\n  What to do: use a model server on this machine (to allow another one anyway, set "proxy.allowRemoteUpstream": true in the config yourself).`,
    proxyLine: (port, up) => `  proxy: http://127.0.0.1:${port}/v1 in front of ${up}`,
    upstreamDown: u => `  note: ${u} does not answer now. Start your model server before you chat, for example:\n`
      + '    llama.cpp:  llama-server -m <model.gguf> --port 8080\n    Ollama:     ollama serve\n    LM Studio:  Developer tab > Start Server',
    checkStep: 'Checking (a throwaway memory; your own is not touched)',
    startingModels: '  starting the models (the first time can take a minute)...',
    recalledOk: '  ✓ a message from one conversation was recalled in another',
    factsOk: '  ✓ facts are extracted',
    timedOut: (what, s, out, hint) => `${what} did not work within ${s} s.\n  What to do: ${hint}\n  The memory's output:\n${out}`,
    whatStart: 'starting the memory', whatIndex: 'storing a message (the search model)', whatFacts: 'fact extraction (the facts model)',
    hintStart: 'another program may use the port; see the output below. Run setup again, and with --report if it repeats.',
    hintIndex: log => `the search model's llama-server did not answer. Its log is ${log}. A llama.cpp built for a GPU it cannot use can fail here: try a CPU build, and pass it with --llama-server.`,
    hintFacts: log => `the facts model is slow or did not start. Its log is ${log}. Without a GPU this can take minutes: run setup again, or use --quick.`,
    recallMissing: json => `recall did not find the stored message:\n${json}\n  What to do: run setup again with --report and send the result.`,
    demoStep: 'What it does (a real run on your machine)',
    demoChat1: 'Chat 1', demoNew: 'New chat', demoYou: 'You:', demoAi: 'AI:',
    demoRecalled: d => `tsuzuki recalled (${d}):`,
    demoWithout: 'The same model without tsuzuki',
    demoSystem: 'Answer in one or two short sentences.',
    demoFirst: 'I just started learning the shamisen, and my teacher is Ms. Sato.',
    demoLater: 'What instrument am I learning, and who is my teacher?',
    demoQuickFirst: 'My dog is called Pochi and I live in Sendai.',
    demoQuickLater: "What is my dog's name?",
    demoQuickNote: '  In a chat, your model reads this before it answers. (The quick setup has no model of its own to show the answer.)',
    demoSkipped: why => `  (the demo was skipped: ${why})`,
    startStep: 'Starting the memory server',
    running: port => `  the memory server is running (port ${port})`,
    runningBg: port => `  ✓ the memory server is running in the background (port ${port})`,
    portTaken: (port, config) => `another program answers on port ${port}, so the memory server cannot use it.\n  What to do: stop that program, or set another "port" in ${config}.`,
    didNotStart: log => `the memory server did not start.\n  What to do: see ${log}; run setup again with --report if it repeats.`,
    noConfig: p => `${p} does not exist: run the setup first.`,
    stopped: 'Stopped the memory server.', notRunning: 'The memory server was not running.',
    couldNotStop: (pid, m) => `Could not stop the memory server (pid ${pid}): ${m}`,
    stoppedLlama: "Stopped the memory's llama-servers.", noLlama: 'No llama-server of the memory was running.',
    autostartQ: '  Start the memory server automatically when you log in? [Y/n]',
    whatAutostart: 'whether to start the memory at login (--autostart)',
    autostartOn: (kind, path) => `  ✓ the memory server starts when you log in (${kind}: ${path})`,
    autostartOff: path => `The memory server no longer starts at login (removed ${path}).`,
    autostartFailed: (cmd, out, path) => `${cmd} failed: ${out}\n  What to do: start the memory by hand with --start, or remove ${path} and try again.`,
    autostartLater: cmd => `  To start it automatically when you log in:  ${cmd} --autostart`,
    doneStep: 'Done',
    doneHermes: yaml => `  Start Hermes. The memory and its models start by themselves when Hermes first needs them.
  Check it with:  hermes memory status
  To let the agent delete memory when you ask, add  forget: true  under  memory: > tsuzuki:  in ${yaml}`,
    doneMcp: (cmd, json) => `  Add the memory to your MCP client. Claude Code:\n    ${cmd}\n  Claude Desktop, Cursor, LM Studio, Jan, Cline... (in their MCP settings: claude_desktop_config.json, mcp.json):\n${json}
  The client starts the memory when it needs it. The model calls recall and remember itself; to let it delete
  memory when you ask, add "--forget" to args.`,
    doneChat: (port, token, up) => `  In your chat app, use this OpenAI-compatible connection instead of your model server:
    URL:      http://127.0.0.1:${port}/v1
    API key:  ${token}
  Every chat is stored and earlier chats are recalled; ${up} still answers.`,
    doneChatStart: cmd => `  After restarting the computer, start the memory again:  ${cmd} --start`,
    doneApp: (cmd, port, config) => `  Start the memory server (keep it running while your program uses it):\n    ${cmd} --start
  Your program sends JSON to http://127.0.0.1:${port} with  Authorization: Bearer <token>  (the token is in ${config}).
  Endpoints: POST /v1/memory/messages, /v1/memory/recall, /v1/memory/search (see the README).`,
    doneMemory: cmd => `  See, search or delete what it keeps, or import your ChatGPT / Claude history:  ${cmd} help`,
    doneStop: cmd => `  The models' llama-servers keep running in the background so the next start is fast (about 0.5 GB of memory,
  4 GB more with facts). Stop everything with:  ${cmd} --stop`,
    reportSaved: p => `The report above is also in ${p}.`,
    reportSend: url => `Paste it into an issue, or send it to the person who asked you to test:\n  ${url}`,
    stuck: url => `\nStuck? Run setup again with --report and paste what it prints into an issue:\n  ${url}`,
    updateStep: v => `Updating tsuzuki ${v}`,
    notGit: url => `this folder is not a git checkout, so it cannot update itself.\n  What to do: download the new version from ${url}.`,
    dirty: files => `files of this checkout were changed here, so git would not update it:\n${files}\n  What to do: undo or commit those changes, then run --update again.`,
    pulling: 'getting the newest version (git pull)',
    upToDate: v => `  tsuzuki ${v} is the newest version.`,
    updated: (a, b) => `  ✓ updated: ${a} → ${b}`,
    restarting: '  restarting the memory server with the new version',
    whatsNew: "What's new:",
    restartHosts: '  Hermes and MCP clients load the memory when they start: restart them to use the new version.',
  },

  ja: {
    help: v => `tsuzuki ${v} のセットアップ。使い方：node scripts/setup.mjs [オプション]

  --for chat|mcp|hermes|app
                         記憶をどこで使うか（無ければ聞く）：
                           chat    自分のモデルのサーバー（llama.cpp、Ollama、LM Studio など）を使うチャットアプリ
                           mcp     MCP クライアント（Claude Desktop、Claude Code、Cursor、LM Studio、Jan、Cline など）
                           hermes  Hermes Agent
                           app     自分で作るプログラム（HTTP）
  --quick                検索のモデルだけ（334 MB、5 分ほど）。事実の抽出なし（--no-facts と同じ）
  --upstream URL         --for chat のとき：モデルのサーバーの OpenAI 互換の URL（既定：llama.cpp :8080、
                         Ollama :11434、LM Studio :1234 のうち最初に応答したもの）
  --profile NAME         Hermes のプロファイル（既定は "default"。Hermes のホームそのもの）
  --hermes-home DIR      Hermes のホーム（既定：$HERMES_HOME、なければ ~/.hermes。Windows は %LOCALAPPDATA%\\hermes）
  --models DIR           モデルの置き場所（既定：~/.tsuzuki/models）
  --llama-server PATH    llama.cpp の llama-server が PATH に無いとき、その場所
  --report               最後に、issue に貼れる動作報告を表示する（失敗したときも出る）
  --lang en|ja           このセットアップの言語（既定：システムの言語）
  --yes                  何も聞かない：既定の答えとモデルのライセンスを受け入れる
  --skip-build           ビルド済みならビルドしない。--skip-check：最後の確認をしない

  --start                記憶のサーバーを裏で起動して終わる
  --stop                 記憶のサーバーと、それが起動した llama-server を止めて終わる
  --autostart[=off]      ログインしたときに記憶のサーバーを起動する（=off でやめる）。設定して終わる
  --update               最新版を取得（git pull）、ビルド、記憶のサーバーを再起動、変更点を表示
  --version, --help`,
    banner: v => `tsuzuki ${v} のセットアップ（Ctrl+C でいつでも止められます。もう一度実行すると続きから）`,
    unknownArg: a => `知らないオプションです：${a}（--help を参照）`,
    needsValue: k => `--${k} には値が要ります`,
    nodeOld: (v, how) => `Node ${v} は古すぎます。tsuzuki には Node 22（22.19 以上）か 24 以上が要ります。\n  入れ方：${how}\n  入れたら、もう一度実行してください。`,
    nodeHow: { darwin: 'brew install node（または https://nodejs.org のインストーラー）', win32: 'winget install OpenJS.NodeJS.LTS（入れたら新しいターミナルを開く。winget は新しいターミナルにしか PATH を反映しない）',
      linux: 'https://nodejs.org のインストーラーかパッケージ（ディストリビューションのパッケージは古いことが多い）' },
    menu: `
記憶をどこで使いますか？ 言葉で 1 つ入力してください：
  chat     自分のモデルのサーバー（llama.cpp、Ollama、LM Studio など）を使うチャットアプリ：どのチャットにも記憶が効く
  mcp      MCP クライアント：Claude Desktop、Claude Code、Cursor、LM Studio、Jan、Cline など
  hermes   Hermes Agent
  app      自分で作るプログラム（127.0.0.1 の HTTP API）
>`,
    whatTarget: '記憶をどこで使うか（--for chat、mcp、hermes、app）',
    menuNumber: '  言葉（chat、mcp、hermes、app）で入力してください。前の版では番号でしたが、番号の意味が変わりました。',
    menuUnknown: a => `  「${a}」はありません。chat、mcp、hermes、app のどれかを入力してください。`,
    forInvalid: v => `--for は chat、mcp、hermes、app のどれかです（「${v}」ではなく）`,
    needFor: '--yes のときは、記憶をどこで使うかを --for chat、mcp、hermes、app で指定してください。',
    planTitle: 'どこまで入れるか',
    planQuick: (size) => `  quick   検索だけ：ダウンロード ${mb(size)}、5 分ほど。記憶は発言そのものから思い出す。`,
    planFull: (size) => `  full    検索と事実：ダウンロード合計 ${gb(size)}、初回は 15〜30 分ほど（ほとんどがダウンロード）。`
      + '\n          事実があると、会話が溜まっても思い出しが落ちにくい。検証したのはこの設定。',
    planQ: 'quick か full を入力 [full]：',
    whatPlan: 'quick か full か（quick は --quick）',
    planUnknown: a => `  「${a}」はどちらでもありません。quick か full を入力してください。`,
    buildStep: 'プラグインと記憶のサーバーをビルド',
    installing: 'パッケージを入れる', building: 'ビルド', packing: 'パック（DeepSeek Harness 用）',
    cmdFailed: (cmd, cwd, out) => `${cmd} が ${cwd} で失敗しました：\n${out}\n  どうするか：ネットワークの接続を確かめて、もう一度実行してください。同じように失敗したら、--report を付けて実行し、結果を送ってください。`,
    llamaStep: 'llama.cpp（llama-server）を探す',
    llamaMissing: (where, how) => `llama-server が${where}見つかりません。先に llama.cpp を入れてください：\n    ${how}\n  入れたらもう一度実行してください（または --llama-server <llama-server の場所> を付ける）。`,
    llamaWhere: g => g ? ` ${g} に` : ' PATH に',
    llamaHow: { darwin: 'brew install llama.cpp', win32: 'winget install llama.cpp（または https://github.com/ggml-org/llama.cpp/releases の Windows 版）。\n    入れたら新しいターミナルを開く（winget は新しいターミナルにしか PATH を反映しない）',
      linux: 'https://github.com/ggml-org/llama.cpp/releases の Linux 版（または brew install llama.cpp）' },
    llamaBad: (bin, why, text) => `${bin} --help が思ったように動きません（${why}）：\n${text}\n  どうするか：新しい llama.cpp（--embedding があるもの）を入れ、その llama-server を --llama-server で指定してください。`,
    modelsStep: dir => `モデル（${dir}）`,
    already: f => `  ${f}：ダウンロード済み`,
    checking: (f, size) => `  ${f}：${mb(size)}、確認中...`,
    license: (title, size, license) => `  ${title}：${mb(size)}。\n  ライセンス：${license}。`,
    downloadQ: '  ダウンロードしますか？',
    whatLicense: 'EmbeddingGemma をダウンロードするか（Gemma の利用規約）',
    noEmbedding: '検索にはこのモデルが要るので、ここで止めました。続けるときは、もう一度実行してください。',
    quickNote: '  quick：事実のモデルは入れません。あとで入れるときは --quick を付けずにもう一度実行してください。',
    httpFailed: (s, url) => `ダウンロードに失敗しました：HTTP ${s}（${url}）\n  どうするか：そのアドレスがブラウザで開けるか確かめて、もう一度実行してください。`,
    downloadStopped: (f, why) => `${f} のダウンロードが止まりました（${why}）。\n  どうするか：ネットワークの接続を確かめて、もう一度実行してください。止まったところから続きます。`,
    badFile: (f, d) => `${f} が想定したファイルではありません（sha256 ${d}）。\n  どうするか：もう一度実行して、ダウンロードし直してください。`,
    diskFull: (dir, need, free) => `${dir} の空き容量が足りません：${gb(need)} 必要、空き ${gb(free)}。\n  どうするか：空きを作るか、--models <フォルダ> でモデルを別の場所に置いてください。`,
    hermesStep: 'Hermes Agent につなぐ',
    profileQ: '  Hermes のプロファイル（"default" は Hermes のホームそのもの）[default]：',
    whatProfile: 'Hermes のプロファイル（--profile）',
    noHermes: p => `${p} が見つかりません。\n  どうするか：このプロファイルで Hermes を一度起動するか、--hermes-home <フォルダ> を付けてください（Hermes 以外なら --for chat、mcp、app）。`,
    quitHermes: '  Hermes が動いていたら終了して、Enter を押してください。',
    whatQuit: 'Hermes が止まっていること',
    installerStopped: out => `Hermes の導入スクリプトが止まりました：\n${out}`,
    noPython: 'Python 3 が見つかりません（Hermes の導入スクリプトに要ります）。\n  どうするか：Python 3 を入れるか、Hermes を更新してください（新しい版は自分の Python を持っています）。',
    memorySettings: (p, port) => `  記憶の設定：${p}（ポート ${port}）`,
    searchLine: '  検索：EmbeddingGemma（必要なときに記憶が起動）',
    factsOn: '  事実：Gemma 4 E2B（必要なときにこのマシンで起動。Hermes 自体がクラウドのモデルでも）',
    factsOff: '  事実：オフ（入れるときは --quick を付けずにもう一度実行）',
    appStep: 'プログラム用に記憶のサーバーを用意', mcpStep: 'MCP クライアント用に記憶を用意',
    chatStep: 'チャットアプリ用に記憶のプロキシを用意',
    settings: (p, port, data) => `  設定：${p}（ポート ${port}。記憶は ${data}）`,
    upstreamStep: 'モデルのサーバー',
    found: (name, url) => `  ${name} が ${url} で見つかりました`,
    upstreamQ: d => `  モデルのサーバーの OpenAI 互換の URL [${d}]：`,
    whatUpstream: 'モデルのサーバーの URL（--upstream）',
    notUrl: u => `URL ではありません：${u}`,
    notLocal: u => `${u} はこのマシンのものではありません。思い出した記憶がそこへ送られてしまいます。\n  どうするか：このマシンのモデルのサーバーを使ってください（それでも別のものを使うなら、設定で "proxy.allowRemoteUpstream": true にする）。`,
    proxyLine: (port, up) => `  プロキシ：http://127.0.0.1:${port}/v1（転送先 ${up}）`,
    upstreamDown: u => `  注意：${u} はいま応答しません。チャットの前にモデルのサーバーを起動してください。例：\n`
      + '    llama.cpp：llama-server -m <モデル.gguf> --port 8080\n    Ollama：ollama serve\n    LM Studio：Developer タブ > Start Server',
    checkStep: '確認（使い捨ての記憶で。あなたの記憶には触りません）',
    startingModels: '  モデルを起動中（初回は 1 分ほどかかることがあります）...',
    recalledOk: '  ✓ ある会話の発言を、別の会話で思い出せた',
    factsOk: '  ✓ 事実を抜き出せた',
    timedOut: (what, s, out, hint) => `${what}が ${s} 秒以内に終わりませんでした。\n  どうするか：${hint}\n  記憶の出力：\n${out}`,
    whatStart: '記憶の起動', whatIndex: '発言の保存（検索のモデル）', whatFacts: '事実の抽出（事実のモデル）',
    hintStart: 'ほかのプログラムがポートを使っているかもしれません。下の出力を見てください。もう一度実行し、繰り返すなら --report を付けてください。',
    hintIndex: log => `検索のモデルの llama-server が応答しませんでした。ログは ${log}。使えない GPU 向けにビルドされた llama.cpp だとここで失敗することがあります。CPU 版を試し、--llama-server で指定してください。`,
    hintFacts: log => `事実のモデルが遅いか、起動しませんでした。ログは ${log}。GPU がないと数分かかることがあります。もう一度実行するか、--quick を使ってください。`,
    recallMissing: json => `保存した発言を思い出せませんでした：\n${json}\n  どうするか：--report を付けてもう一度実行し、結果を送ってください。`,
    demoStep: '何ができるか（あなたのマシンでの実際の動作）',
    demoChat1: 'チャット 1', demoNew: '新しいチャット', demoYou: 'あなた：', demoAi: 'AI：',
    demoRecalled: d => `tsuzuki が思い出した記憶（${d}）：`,
    demoWithout: '同じモデルで tsuzuki なし',
    demoSystem: '1〜2文で短く答えてください。',
    demoFirst: '最近、三味線を習い始めたんだ。先生は佐藤さん。',
    demoLater: '私が習ってる楽器と、先生の名前を覚えてる？',
    demoQuickFirst: 'うちの犬の名前はポチで、仙台に住んでるよ。',
    demoQuickLater: 'うちの犬の名前、覚えてる？',
    demoQuickNote: '  チャットでは、モデルが答える前にこれを読みます（quick には答えて見せるモデルがありません）。',
    demoSkipped: why => `  （デモは飛ばしました：${why}）`,
    startStep: '記憶のサーバーを起動',
    running: port => `  記憶のサーバーは動いています（ポート ${port}）`,
    runningBg: port => `  ✓ 記憶のサーバーが裏で動いています（ポート ${port}）`,
    portTaken: (port, config) => `ポート ${port} でほかのプログラムが応答しているので、記憶のサーバーが使えません。\n  どうするか：そのプログラムを止めるか、${config} の "port" を変えてください。`,
    didNotStart: log => `記憶のサーバーが起動しませんでした。\n  どうするか：${log} を見てください。繰り返すなら --report を付けてもう一度実行してください。`,
    noConfig: p => `${p} がありません。先にセットアップを実行してください。`,
    stopped: '記憶のサーバーを止めました。', notRunning: '記憶のサーバーは動いていませんでした。',
    couldNotStop: (pid, m) => `記憶のサーバー（pid ${pid}）を止められませんでした：${m}`,
    stoppedLlama: '記憶の llama-server を止めました。', noLlama: '記憶の llama-server は動いていませんでした。',
    autostartQ: '  ログインしたときに記憶のサーバーを自動で起動しますか？ [Y/n]',
    whatAutostart: 'ログイン時に起動するか（--autostart）',
    autostartOn: (kind, path) => `  ✓ ログインしたときに記憶のサーバーが起動します（${kind}：${path}）`,
    autostartOff: path => `ログイン時の起動をやめました（${path} を消しました）。`,
    autostartFailed: (cmd, out, path) => `${cmd} が失敗しました：${out}\n  どうするか：--start で手で起動するか、${path} を消してもう一度試してください。`,
    autostartLater: cmd => `  ログインしたときに自動で起動するには：${cmd} --autostart`,
    doneStep: '完了',
    doneHermes: yaml => `  Hermes を起動してください。記憶とモデルは、Hermes が最初に必要としたときに自分で起動します。
  確認：hermes memory status
  頼んだときに記憶を消せるようにするには、${yaml} の  memory: > tsuzuki:  の下に  forget: true  を足してください。`,
    doneMcp: (cmd, json) => `  MCP クライアントに記憶を足してください。Claude Code：\n    ${cmd}\n  Claude Desktop、Cursor、LM Studio、Jan、Cline など（MCP の設定：claude_desktop_config.json、mcp.json）：\n${json}
  記憶は、クライアントが必要としたときに起動します。recall（思い出す）と remember（覚える）はモデルが自分で呼びます。
  頼んだときに記憶を消せるようにするには、args に "--forget" を足してください。`,
    doneChat: (port, token, up) => `  チャットアプリで、モデルのサーバーの代わりにこの OpenAI 互換の接続を使ってください：
    URL：     http://127.0.0.1:${port}/v1
    API キー：${token}
  チャットはすべて保存され、前のチャットから思い出します。答えるのは今までどおり ${up} です。`,
    doneChatStart: cmd => `  パソコンを再起動したあとは、記憶をもう一度起動してください：${cmd} --start`,
    doneApp: (cmd, port, config) => `  記憶のサーバーを起動してください（プログラムが使う間は動かしたままに）：\n    ${cmd} --start
  プログラムは http://127.0.0.1:${port} に JSON を送ります。ヘッダー  Authorization: Bearer <トークン>（トークンは ${config}）。
  エンドポイント：POST /v1/memory/messages、/v1/memory/recall、/v1/memory/search（README を参照）。`,
    doneMemory: cmd => `  覚えていることを見る・探す・消す、ChatGPT や Claude の履歴を取り込む：${cmd} help`,
    doneStop: cmd => `  モデルの llama-server は、次にすぐ起動できるよう裏で動いたままです（メモリ約 0.5 GB、事実ありでさらに約 4 GB）。
  全部止めるとき：${cmd} --stop`,
    reportSaved: p => `上の報告は ${p} にも保存しました。`,
    reportSend: url => `issue に貼るか、テストを頼んだ人に送ってください：\n  ${url}`,
    stuck: url => `\nうまくいかないときは、--report を付けてもう一度実行し、表示された内容を issue に貼ってください：\n  ${url}`,
    updateStep: v => `tsuzuki ${v} を更新`,
    notGit: url => `このフォルダは git で取ってきたものではないので、自分で更新できません。\n  どうするか：${url} から新しい版をダウンロードしてください。`,
    dirty: files => `このフォルダのファイルが書き換えられているので、git が更新しません：\n${files}\n  どうするか：その変更を元に戻すかコミットしてから、もう一度 --update を実行してください。`,
    pulling: '最新版を取得（git pull）',
    upToDate: v => `  tsuzuki ${v} が最新版です。`,
    updated: (a, b) => `  ✓ 更新しました：${a} → ${b}`,
    restarting: '  新しい版で記憶のサーバーを再起動',
    whatsNew: '変更点：',
    restartHosts: '  Hermes と MCP クライアントは、起動したときに記憶を読み込みます。新しい版を使うには再起動してください。',
  },
}
