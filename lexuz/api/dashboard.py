"""
lexuz.api.dashboard

Веб-интерфейс (Web Dashboard) для Legal RAG Lex.uz:
- Интерактивный юридический поисковый запрос (Sandbox)
- Пакетное тестирование (Batch Evaluator)
- Просмотр подтверждающих контекстов и цитат с прямыми ссылками на Lex.uz
- Мониторинг здоровья системы и версий компонентов (/v1/meta, /v1/health)
"""

from fastapi.responses import HTMLResponse

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="uz" class="h-full bg-slate-900 text-slate-100">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Lex.uz Legal RAG Platform</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    body { font-family: 'Plus Jakarta Sans', sans-serif; }
    code, pre { font-family: 'JetBrains Mono', monospace; }
    .glass-card {
      background: rgba(30, 41, 59, 0.7);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .citation-pill {
      display: inline-flex;
      align-items: center;
      padding: 0.1rem 0.45rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 700;
      background: rgba(14, 165, 233, 0.2);
      color: #38bdf8;
      border: 1px solid rgba(56, 189, 248, 0.4);
      margin: 0 0.15rem;
      cursor: pointer;
    }
    .markdown-content {
      color: #e2e8f0;
      line-height: 1.8 !important;
      font-size: 0.95rem;
    }
    .markdown-content p {
      margin-bottom: 1.15rem !important;
      line-height: 1.8 !important;
    }
    .markdown-content p:last-child {
      margin-bottom: 0 !important;
    }
    .markdown-content strong, .markdown-content b {
      color: #38bdf8 !important;
      font-weight: 700 !important;
    }
    .markdown-content em, .markdown-content i {
      color: #cbd5e1;
      font-style: italic;
    }
    .markdown-content ul {
      list-style-type: disc !important;
      margin-left: 1.75rem !important;
      margin-top: 0.6rem !important;
      margin-bottom: 1.15rem !important;
    }
    .markdown-content ol {
      list-style-type: decimal !important;
      margin-left: 1.75rem !important;
      margin-top: 0.6rem !important;
      margin-bottom: 1.15rem !important;
    }
    .markdown-content li {
      margin-bottom: 0.45rem !important;
      line-height: 1.7 !important;
    }
    .markdown-content h1, .markdown-content h2, .markdown-content h3, .markdown-content h4 {
      color: #38bdf8 !important;
      font-weight: 700 !important;
      margin-top: 1.25rem !important;
      margin-bottom: 0.6rem !important;
    }
    .markdown-content blockquote {
      border-left: 3px solid #38bdf8 !important;
      padding-left: 0.85rem !important;
      margin: 1rem 0 !important;
      color: #94a3b8 !important;
      font-style: italic;
      background: rgba(56, 189, 248, 0.05);
      border-radius: 0 0.5rem 0.5rem 0;
      padding-top: 0.35rem !important;
      padding-bottom: 0.35rem !important;
    }
  </style>
</head>
<body class="h-full flex flex-col overflow-x-hidden">
  <!-- Navbar -->
  <header class="glass-card sticky top-0 z-50 border-b border-slate-800">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center text-white font-black text-xl shadow-lg shadow-sky-500/20">
          <i class="fa-solid fa-scale-balanced"></i>
        </div>
        <div>
          <h1 class="text-lg font-bold text-white tracking-tight flex items-center gap-2">
            Lex.uz <span class="text-xs px-2 py-0.5 rounded bg-sky-500/20 text-sky-400 font-semibold border border-sky-500/30">Legal RAG v1</span>
          </h1>
          <p class="text-xs text-slate-400">O'zbekiston Qonunchiligi AI Tahlil Tizimi</p>
        </div>
      </div>

      <!-- System status pills & API Key -->
      <div class="flex items-center space-x-4">
        <div id="healthBadge" class="hidden sm:flex items-center space-x-2 px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs font-medium">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span id="healthText">Tizim ishlamoqda</span>
        </div>
        
        <div class="flex items-center bg-slate-800/80 rounded-lg p-1 border border-slate-700">
          <i class="fa-solid fa-key text-xs text-slate-400 ml-2 mr-1"></i>
          <input id="apiKeyInput" type="password" placeholder="API Key..." class="bg-transparent text-xs text-slate-200 px-2 py-1 outline-none w-28 sm:w-36" value="lexuz-test-api-key" />
          <button onclick="saveApiKey()" class="text-xs bg-sky-600 hover:bg-sky-500 text-white px-2 py-1 rounded transition">Saqlash</button>
        </div>

        <a href="/docs" target="_blank" class="text-xs text-slate-400 hover:text-white px-2 py-1 rounded hover:bg-slate-800 flex items-center gap-1 transition">
          <i class="fa-solid fa-book-open"></i> <span class="hidden md:inline">OpenAPI Docs</span>
        </a>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
    
    <!-- Meta Cards Banner -->
    <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
      <div class="glass-card rounded-xl p-3 border border-slate-800">
        <p class="text-[10px] text-slate-400 font-bold uppercase tracking-wider">LLM Model</p>
        <p id="metaLLM" class="text-xs font-semibold text-sky-400 mt-1 truncate">gemini-2.5-flash</p>
      </div>
      <div class="glass-card rounded-xl p-3 border border-slate-800">
        <p class="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Embedder</p>
        <p id="metaEmb" class="text-xs font-semibold text-indigo-400 mt-1 truncate">BAAI/bge-m3</p>
      </div>
      <div class="glass-card rounded-xl p-3 border border-slate-800">
        <p class="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Reranker</p>
        <p id="metaRerank" class="text-xs font-semibold text-purple-400 mt-1 truncate">bge-reranker-v2-m3</p>
      </div>
      <div class="glass-card rounded-xl p-3 border border-slate-800">
        <p class="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Qdrant Indeks</p>
        <p id="metaIndex" class="text-xs font-semibold text-emerald-400 mt-1 truncate">lex-idx-2026-10-28</p>
      </div>
      <div class="glass-card rounded-xl p-3 border border-slate-800">
        <p class="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Snapshot Sanasi</p>
        <p id="metaDate" class="text-xs font-semibold text-amber-400 mt-1">2026-10-28</p>
      </div>
      <div class="glass-card rounded-xl p-3 border border-slate-800">
        <p class="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Qdrant Vektorlar</p>
        <p id="metaVectors" class="text-xs font-semibold text-rose-400 mt-1">~1,176,871</p>
      </div>
    </div>

    <!-- Tabs Navigation -->
    <div class="flex items-center space-x-2 border-b border-slate-800 pb-3">
      <button onclick="switchTab('single')" id="tabBtnSingle" class="px-4 py-2 text-sm font-semibold rounded-lg bg-sky-600 text-white shadow-lg shadow-sky-600/20 transition flex items-center gap-2">
        <i class="fa-solid fa-magnifying-glass"></i> Interaktiv So'rov (Sandbox)
      </button>
      <button onclick="switchTab('batch')" id="tabBtnBatch" class="px-4 py-2 text-sm font-semibold rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition flex items-center gap-2">
        <i class="fa-solid fa-layer-group"></i> Batch Baholash (10 ta savol)
      </button>
      <button onclick="switchTab('history')" id="tabBtnHistory" class="px-4 py-2 text-sm font-semibold rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition flex items-center gap-2">
        <i class="fa-solid fa-clock-rotate-left"></i> Batch Tarixi (30 kun)
      </button>
    </div>

    <!-- TAB 1: Single Query Sandbox -->
    <section id="tabSingle" class="space-y-6">
      <div class="glass-card rounded-2xl p-6 border border-slate-800 shadow-xl space-y-4">
        <!-- Mode selector: Simple vs Formal -->
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3 rounded-xl bg-slate-950/60 border border-slate-800">
          <div class="flex items-center gap-2">
            <span class="text-xs text-slate-400 font-bold uppercase tracking-wider">Javob rejimi:</span>
            <div class="inline-flex p-1 bg-slate-900 rounded-xl border border-slate-700/80">
              <button type="button" id="btnStyleSimple" onclick="setModeStyle('simple')" class="px-3.5 py-1.5 text-xs font-bold rounded-lg transition flex items-center gap-1.5 bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 shadow-sm">
                <span>🟢</span> Oddiy tilda (Tushunarli)
              </button>
              <button type="button" id="btnStyleFormal" onclick="setModeStyle('formal')" class="px-3.5 py-1.5 text-xs font-semibold text-slate-400 hover:text-white rounded-lg transition flex items-center gap-1.5">
                <span>⚖️</span> Rasmiy tilda (Yuridik)
              </button>
            </div>
            <input type="hidden" id="selectedStyle" value="simple" />
          </div>
          <div class="text-[11px] text-slate-400 flex items-center gap-1">
            <i class="fa-solid fa-circle-info text-sky-400"></i>
            <span id="styleHint">Oddiy fuqarolar uchun sodda, tushunarli va qulay leksikonda tushuntirish</span>
          </div>
        </div>

        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <label class="text-sm font-bold text-slate-200 flex items-center gap-2">
            <i class="fa-regular fa-comment-dots text-sky-400"></i> Savolingizni kiriting:
          </label>
          <div class="flex flex-wrap gap-2">
            <button onclick="fillSample(1)" class="text-[11px] bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1 rounded-md border border-slate-700 transition">
              🇺🇿 Ta'til muddati (Lotin)
            </button>
            <button onclick="fillSample(4)" class="text-[11px] bg-emerald-950/50 hover:bg-emerald-900/60 text-emerald-300 px-2.5 py-1 rounded-md border border-emerald-700/60 transition">
              🗣️ Otpuska necha kun ozi (Oddiy)
            </button>
            <button onclick="fillSample(5)" class="text-[11px] bg-emerald-950/50 hover:bg-emerald-900/60 text-emerald-300 px-2.5 py-1 rounded-md border border-emerald-700/60 transition">
              🗣️ Ishdan haydashsa nima qilaman
            </button>
            <button onclick="fillSample(2)" class="text-[11px] bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1 rounded-md border border-slate-700 transition">
              🇷🇺 Срок давности (Русский)
            </button>
            <button onclick="fillSample(3)" class="text-[11px] bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1 rounded-md border border-slate-700 transition">
              ❌ Javobsiz savol testi
            </button>
          </div>
        </div>

        <textarea id="singleQuestion" rows="3" class="w-full bg-slate-950/80 text-white rounded-xl p-4 border border-slate-700 focus:border-sky-500 focus:ring-1 focus:ring-sky-500 outline-none text-sm leading-relaxed" placeholder="Masalan: Yillik asosiy mehnat ta'tilining eng kam muddati qancha?"></textarea>

        <!-- Options row -->
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-2">
          <div>
            <label class="text-xs text-slate-400 font-medium">Top-K kontekstlar:</label>
            <select id="singleTopK" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 mt-1">
              <option value="3">3 ta fragment</option>
              <option value="5" selected>5 ta fragment (Default)</option>
              <option value="10">10 ta fragment</option>
            </select>
          </div>
          <div>
            <label class="text-xs text-slate-400 font-medium">Javob tili:</label>
            <select id="singleLang" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 mt-1">
              <option value="auto" selected>Auto (Savol tilida)</option>
              <option value="uz-Latn">O'zbekcha (Lotin)</option>
              <option value="uz-Cyrl">Ўзбекча (Кирилл)</option>
              <option value="ru">Русский</option>
            </select>
          </div>
          <div>
            <label class="text-xs text-slate-400 font-medium">Temperature:</label>
            <input id="singleTemp" type="number" step="0.1" min="0" max="1" value="0.0" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 mt-1" />
          </div>
          <div class="flex items-end">
            <button onclick="executeSingleQuery()" id="singleSubmitBtn" class="w-full bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 text-white font-bold py-2 px-4 rounded-xl shadow-lg shadow-sky-500/20 text-sm transition flex items-center justify-center gap-2">
              <i class="fa-solid fa-paper-plane"></i> Izlash & Javob
            </button>
          </div>
        </div>
      </div>

      <!-- Result Card -->
      <div id="singleResultContainer" class="hidden space-y-6">
        <div class="glass-card rounded-2xl p-6 border border-slate-800 shadow-xl space-y-4">
          <div class="flex items-center justify-between border-b border-slate-800 pb-4">
            <div class="flex items-center gap-3">
              <span id="resStatusBadge" class="px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                OK
              </span>
              <span id="resStyleBadge" class="px-2.5 py-1 rounded-full text-xs font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                🟢 Oddiy tilda
              </span>
              <span id="resTimingBadge" class="text-xs text-slate-400">
                <i class="fa-regular fa-clock mr-1"></i> <span id="resTotalMs">0</span> ms
              </span>
              <span id="resTokensBadge" class="text-xs text-slate-400">
                <i class="fa-solid fa-microchip mr-1"></i> <span id="resTokens">0</span> tokens
              </span>
            </div>
            <button onclick="copyResultJson()" class="text-xs text-slate-400 hover:text-white px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 transition flex items-center gap-1">
              <i class="fa-regular fa-copy"></i> JSON nusxalash
            </button>
          </div>

          <!-- Rewritten Query Box (if formalized) -->
          <div id="resRewrittenQueryBox" class="hidden p-3.5 rounded-xl bg-sky-950/40 border border-sky-800/60 text-xs space-y-1">
            <span class="font-bold text-sky-400 flex items-center gap-1.5">
              <i class="fa-solid fa-wand-magic-sparkles"></i> Qonunchilik bo'yicha rasmiylashtirilgan yuridik so'rov (Legal Search Query):
            </span>
            <p id="resRewrittenQueryText" class="text-slate-200 font-mono pl-4 border-l-2 border-sky-500/50"></p>
          </div>

          <!-- Answer Text Area -->
          <div class="space-y-2">
            <h3 class="text-xs font-bold text-slate-400 uppercase tracking-wider">AI Yuridik Xulosasi:</h3>
            <div id="resAnswerText" class="p-5 rounded-xl bg-slate-950/60 border border-slate-800 text-slate-100 text-sm leading-relaxed font-normal markdown-content"></div>
          </div>

          <!-- Citations Section -->
          <div id="citationsSection" class="space-y-3 pt-2">
            <h3 class="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
              <i class="fa-solid fa-book-bookmark text-sky-400"></i> Havola qilingan manbalar (Citations):
            </h3>
            <div id="citationsList" class="grid grid-cols-1 md:grid-cols-2 gap-3"></div>
          </div>

          <!-- Contexts Section (Accordion) -->
          <div class="pt-4 border-t border-slate-800">
            <details class="group">
              <summary class="cursor-pointer text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center justify-between py-2 group-open:text-sky-400">
                <span><i class="fa-solid fa-cubes-stacked mr-1"></i> Topilgan qonun bo'laklari (Contexts: <span id="contextsCount">0</span> ta)</span>
                <i class="fa-solid fa-chevron-down group-open:rotate-180 transition"></i>
              </summary>
              <div id="contextsList" class="space-y-3 mt-3"></div>
            </details>
          </div>
        </div>
      </div>
    </section>

    <!-- TAB 2: Batch Evaluator -->
    <section id="tabBatch" class="hidden space-y-6">
      <div class="glass-card rounded-2xl p-6 border border-slate-800 shadow-xl space-y-4">
        <div class="flex items-center justify-between">
          <div>
            <h2 class="text-base font-bold text-white">Batch Evaluator (POST /v1/rag/batch)</h2>
            <p class="text-xs text-slate-400">Spetsifikatsiyadagi b01 – b10 test paketlarini yuborish va tekshirish (1–10 savol)</p>
          </div>
          <button onclick="loadSampleBatch()" class="text-xs bg-slate-800 hover:bg-slate-700 text-sky-400 px-3 py-1.5 rounded-lg border border-slate-700 transition">
            <i class="fa-solid fa-wand-magic-sparkles mr-1"></i> Namuna yuklash (b01)
          </button>
        </div>

        <div class="grid grid-cols-1 sm:grid-cols-4 gap-3">
          <div>
            <label class="text-xs text-slate-400">run_id:</label>
            <input id="batchRunId" type="text" value="eval-2026-11-03" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-xs text-white mt-1" />
          </div>
          <div>
            <label class="text-xs text-slate-400">batch_id:</label>
            <input id="batchBatchId" type="text" value="eval-2026-11-03-b01" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-xs text-white mt-1" />
          </div>
          <div>
            <label class="text-xs text-slate-400">batch_index / total:</label>
            <div class="flex items-center gap-2 mt-1">
              <input id="batchIndex" type="number" value="1" min="1" max="10" class="w-1/2 bg-slate-900 border border-slate-700 rounded-lg p-2 text-xs text-white" />
              <span class="text-slate-500">/</span>
              <input id="batchTotal" type="number" value="10" min="1" max="100" class="w-1/2 bg-slate-900 border border-slate-700 rounded-lg p-2 text-xs text-white" />
            </div>
          </div>
          <div>
            <label class="text-xs text-slate-400">Javob uslubi (style):</label>
            <select id="batchStyle" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-xs text-white mt-1">
              <option value="simple" selected>🟢 Oddiy tilda (Simple)</option>
              <option value="formal">⚖️ Rasmiy tilda (Formal)</option>
            </select>
          </div>
        </div>

        <div>
          <label class="text-xs text-slate-400">Batch JSON So'rovi (items massivi):</label>
          <textarea id="batchJsonArea" rows="8" class="w-full bg-slate-950/80 text-emerald-400 font-mono text-xs rounded-xl p-4 border border-slate-700 mt-1 outline-none"></textarea>
        </div>

        <button onclick="executeBatchQuery()" id="batchSubmitBtn" class="bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-2.5 px-6 rounded-xl shadow-lg shadow-indigo-600/20 text-sm transition flex items-center gap-2">
          <i class="fa-solid fa-play"></i> Batchni yuborish
        </button>
      </div>

      <!-- Batch Results Table -->
      <div id="batchResultContainer" class="hidden glass-card rounded-2xl p-6 border border-slate-800 shadow-xl space-y-4">
        <div class="flex items-center justify-between border-b border-slate-800 pb-4">
          <div class="flex items-center gap-4">
            <h3 class="text-sm font-bold text-white">Batch natijalari</h3>
            <div class="flex items-center gap-2 text-xs">
              <span id="batchSummaryTotal" class="px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-medium">Jami: 0</span>
              <span id="batchSummaryOk" class="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-medium">OK: 0</span>
              <span id="batchSummaryNoAns" class="px-2 py-0.5 rounded bg-amber-500/20 text-amber-400 font-medium">No-Ans: 0</span>
              <span id="batchSummaryErr" class="px-2 py-0.5 rounded bg-rose-500/20 text-rose-400 font-medium">Error: 0</span>
            </div>
          </div>
          <span id="batchTotalTime" class="text-xs text-slate-400"></span>
        </div>

        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs text-slate-300">
            <thead class="text-slate-500 uppercase bg-slate-950/50 text-[10px]">
              <tr>
                <th class="p-3">ID</th>
                <th class="p-3">Status</th>
                <th class="p-3">Javob</th>
                <th class="p-3">Manbalar (Citations)</th>
                <th class="p-3">Vaqt (ms)</th>
              </tr>
            </thead>
            <tbody id="batchTableBody" class="divide-y divide-slate-800/60"></tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- TAB 3: Batch History -->
    <section id="tabHistory" class="hidden space-y-6">
      <div class="glass-card rounded-2xl p-6 border border-slate-800 shadow-xl space-y-4">
        <div class="flex items-center justify-between">
          <div>
            <h2 class="text-base font-bold text-white">Bajarilgan Batchlar Tarixi (30 kun saqlash)</h2>
            <p class="text-xs text-slate-400">SQLite ma'lumotlar bazasidagi saqlangan natijalar</p>
          </div>
          <button onclick="loadHistory()" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700 transition">
            <i class="fa-solid fa-arrows-rotate mr-1"></i> Yangilash
          </button>
        </div>

        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs text-slate-300">
            <thead class="text-slate-500 uppercase bg-slate-950/50 text-[10px]">
              <tr>
                <th class="p-3">Batch ID</th>
                <th class="p-3">Run ID</th>
                <th class="p-3">Status</th>
                <th class="p-3">Savollar (OK/No/Err)</th>
                <th class="p-3">Vaqt (ms)</th>
                <th class="p-3">Yaratilgan</th>
                <th class="p-3">Harakat</th>
              </tr>
            </thead>
            <tbody id="historyTableBody" class="divide-y divide-slate-800/60"></tbody>
          </table>
        </div>
      </div>
    </section>

  </main>

  <!-- Footer -->
  <footer class="border-t border-slate-800 py-6 text-center text-xs text-slate-500">
    <p>Lex.uz Legal RAG API Engine · O'zbekiston Respublikasi Qonunchilik Bazasiga asoslangan</p>
  </footer>

  <script>
    let activeResultData = null;
    let currentStyle = "simple";

    function setModeStyle(style) {
      currentStyle = style;
      const hiddenInput = document.getElementById("selectedStyle");
      if (hiddenInput) hiddenInput.value = style;
      const btnSimple = document.getElementById("btnStyleSimple");
      const btnFormal = document.getElementById("btnStyleFormal");
      const hint = document.getElementById("styleHint");

      if (style === "simple") {
        if (btnSimple) btnSimple.className = "px-3.5 py-1.5 text-xs font-bold rounded-lg transition flex items-center gap-1.5 bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 shadow-sm";
        if (btnFormal) btnFormal.className = "px-3.5 py-1.5 text-xs font-semibold text-slate-400 hover:text-white rounded-lg transition flex items-center gap-1.5";
        if (hint) hint.innerText = "Oddiy fuqarolar uchun sodda, tushunarli va qulay leksikonda tushuntirish";
      } else {
        if (btnFormal) btnFormal.className = "px-3.5 py-1.5 text-xs font-bold rounded-lg transition flex items-center gap-1.5 bg-indigo-500/20 text-indigo-400 border border-indigo-500/40 shadow-sm";
        if (btnSimple) btnSimple.className = "px-3.5 py-1.5 text-xs font-semibold text-slate-400 hover:text-white rounded-lg transition flex items-center gap-1.5";
        if (hint) hint.innerText = "Yuristlar va rasmiy xulosalar uchun qat'iy yuridik terminologiya bilan tahlil";
      }
    }

    function getApiKey() {
      return localStorage.getItem("lexuz_api_key") || document.getElementById("apiKeyInput").value || "lexuz-test-api-key";
    }

    function saveApiKey() {
      const val = document.getElementById("apiKeyInput").value.trim();
      localStorage.setItem("lexuz_api_key", val);
      alert("API kalit saqlandi: " + val);
    }

    // Init state
    window.addEventListener("DOMContentLoaded", () => {
      const stored = localStorage.getItem("lexuz_api_key");
      if (stored) {
        document.getElementById("apiKeyInput").value = stored;
      }
      fetchMetaAndHealth();
      loadSampleBatch();
    });

    async function fetchMetaAndHealth() {
      try {
        const [metaRes, healthRes] = await Promise.all([
          fetch("/v1/meta"),
          fetch("/v1/health")
        ]);
        if (metaRes.ok) {
          const meta = await metaRes.json();
          document.getElementById("metaLLM").innerText = meta.llm_model;
          document.getElementById("metaEmb").innerText = meta.embedding_model;
          document.getElementById("metaRerank").innerText = meta.reranker;
          document.getElementById("metaIndex").innerText = meta.index_version;
          document.getElementById("metaDate").innerText = meta.lex_snapshot_date;
        }
        if (healthRes.ok) {
          const health = await healthRes.json();
          const badge = document.getElementById("healthBadge");
          badge.classList.remove("hidden");
          if (health.qdrant && health.qdrant.points_count) {
            document.getElementById("metaVectors").innerText = health.qdrant.points_count.toLocaleString();
          }
        }
      } catch (e) {
        console.error("Meta load error", e);
      }
    }

    function switchTab(tab) {
      document.getElementById("tabSingle").classList.add("hidden");
      document.getElementById("tabBatch").classList.add("hidden");
      document.getElementById("tabHistory").classList.add("hidden");

      document.getElementById("tabBtnSingle").className = "px-4 py-2 text-sm font-semibold rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition flex items-center gap-2";
      document.getElementById("tabBtnBatch").className = "px-4 py-2 text-sm font-semibold rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition flex items-center gap-2";
      document.getElementById("tabBtnHistory").className = "px-4 py-2 text-sm font-semibold rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition flex items-center gap-2";

      if (tab === "single") {
        document.getElementById("tabSingle").classList.remove("hidden");
        document.getElementById("tabBtnSingle").className = "px-4 py-2 text-sm font-semibold rounded-lg bg-sky-600 text-white shadow-lg shadow-sky-600/20 transition flex items-center gap-2";
      } else if (tab === "batch") {
        document.getElementById("tabBatch").classList.remove("hidden");
        document.getElementById("tabBtnBatch").className = "px-4 py-2 text-sm font-semibold rounded-lg bg-sky-600 text-white shadow-lg shadow-sky-600/20 transition flex items-center gap-2";
      } else if (tab === "history") {
        document.getElementById("tabHistory").classList.remove("hidden");
        document.getElementById("tabBtnHistory").className = "px-4 py-2 text-sm font-semibold rounded-lg bg-sky-600 text-white shadow-lg shadow-sky-600/20 transition flex items-center gap-2";
        loadHistory();
      }
    }

    function fillSample(id) {
      const q = document.getElementById("singleQuestion");
      if (id === 1) {
        q.value = "Yillik asosiy mehnat ta'tilining eng kam muddati qancha?";
        setModeStyle("formal");
        document.getElementById("singleLang").value = "uz-Latn";
      } else if (id === 2) {
        q.value = "Какой срок исковой давности установлен Гражданским кодексом?";
        setModeStyle("formal");
        document.getElementById("singleLang").value = "ru";
      } else if (id === 4) {
        q.value = "otpuska qancha kun beriladi ozi";
        setModeStyle("simple");
        document.getElementById("singleLang").value = "uz-Latn";
      } else if (id === 5) {
        q.value = "ishdan haydab yuborsa nima qilaman";
        setModeStyle("simple");
        document.getElementById("singleLang").value = "uz-Latn";
      } else {
        q.value = "Kosmik kemalarni shaxsiy xususiylashtirish tartibi haqida qonunda nima deyilgan?";
        document.getElementById("singleLang").value = "auto";
      }
    }

    async function executeSingleQuery() {
      const question = document.getElementById("singleQuestion").value.trim();
      if (!question) return alert("Savol matnini kiriting!");

      const topK = parseInt(document.getElementById("singleTopK").value, 10);
      const lang = document.getElementById("singleLang").value;
      const temp = parseFloat(document.getElementById("singleTemp").value);
      const submitBtn = document.getElementById("singleSubmitBtn");

      submitBtn.disabled = true;
      submitBtn.innerHTML = `<i class="fa-solid fa-spinner animate-spin"></i> Qidirilmoqda...`;

      try {
        const res = await fetch("/v1/rag/query", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + getApiKey(),
          },
          body: JSON.stringify({
            question: question,
            options: {
              top_k: topK,
              answer_language: lang,
              temperature: temp,
              style: currentStyle,
              return_contexts: true,
              return_debug: true
            }
          })
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || ("HTTP Xatolik: " + res.status));
        }

        const data = await res.json();
        renderSingleResult(data.result);
      } catch (err) {
        alert("Xatolik: " + err.message);
      } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `<i class="fa-solid fa-paper-plane"></i> Izlash & Javob`;
      }
    }

    function renderSingleResult(item) {
      activeResultData = item;
      const container = document.getElementById("singleResultContainer");
      container.classList.remove("hidden");

      // Badges
      const statusBadge = document.getElementById("resStatusBadge");
      statusBadge.innerText = item.status;
      if (item.status === "ok") {
        statusBadge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
      } else if (item.status === "no_answer") {
        statusBadge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-amber-500/20 text-amber-400 border border-amber-500/30";
      } else {
        statusBadge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-rose-500/20 text-rose-400 border border-rose-500/30";
      }

      // Style badge
      const styleBadge = document.getElementById("resStyleBadge");
      if (styleBadge) {
        if (currentStyle === "formal") {
          styleBadge.className = "px-2.5 py-1 rounded-full text-xs font-bold bg-indigo-500/10 text-indigo-400 border border-indigo-500/20";
          styleBadge.innerText = "⚖️ Rasmiy tilda";
        } else {
          styleBadge.className = "px-2.5 py-1 rounded-full text-xs font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20";
          styleBadge.innerText = "🟢 Oddiy tilda";
        }
      }

      // Rewritten query banner (if formalized and differs from question)
      const rwBox = document.getElementById("resRewrittenQueryBox");
      const rawQ = document.getElementById("singleQuestion").value.trim();
      if (rwBox) {
        if (item.debug && item.debug.rewritten_query && item.debug.rewritten_query.toLowerCase() !== rawQ.toLowerCase()) {
          rwBox.classList.remove("hidden");
          document.getElementById("resRewrittenQueryText").innerText = item.debug.rewritten_query;
        } else {
          rwBox.classList.add("hidden");
        }
      }

      document.getElementById("resTotalMs").innerText = item.timing ? item.timing.total_ms : 0;
      document.getElementById("resTokens").innerText = item.usage ? (item.usage.input_tokens + item.usage.output_tokens) : 0;

      // Answer text with Markdown rendering and highlighted citations
      let rawAnswer = item.answer || "Javob yo'q";
      let formattedAnswer = "";
      if (window.marked && typeof window.marked.parse === "function") {
        try {
          formattedAnswer = window.marked.parse(rawAnswer, { breaks: true, gfm: true });
        } catch (e) {
          console.warn("marked error:", e);
        }
      }
      if (!formattedAnswer) {
        // Robust fallback markdown parser
        let html = rawAnswer.trim();
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong class="text-sky-300 font-bold">$1</strong>');
        html = html.replace(/__(.*?)__/g, '<strong class="text-sky-300 font-bold">$1</strong>');
        html = html.replace(/\*(.*?)\*/g, '<em class="italic text-slate-300">$1</em>');
        html = html.replace(/^[\*\-]\s+(.*$)/gim, '<li class="ml-5 list-disc my-1">$1</li>');
        html = html.replace(/^\d+\.\s+(.*$)/gim, '<li class="ml-5 list-decimal my-1">$1</li>');
        let paragraphs = html.split(/\n\s*\n/);
        formattedAnswer = paragraphs.map(p => {
          p = p.trim();
          if (p.startsWith("<li")) return `<ul class="my-2">${p}</ul>`;
          return `<p class="mb-4 leading-relaxed">${p.replace(/\n/g, '<br/>')}</p>`;
        }).join('');
      }
      formattedAnswer = formattedAnswer.replace(/\[(\d+)\]/g, '<span class="citation-pill">[$1]</span>');
      document.getElementById("resAnswerText").innerHTML = formattedAnswer;

      // Citations
      const citList = document.getElementById("citationsList");
      citList.innerHTML = "";
      if (item.citations && item.citations.length > 0) {
        item.citations.forEach(c => {
          const card = document.createElement("div");
          card.className = "p-3 rounded-xl bg-slate-900 border border-slate-800 space-y-1.5";
          card.innerHTML = `
            <div class="flex items-center justify-between">
              <span class="text-xs font-bold text-sky-400">[${c.ref}] ${c.article || ''}</span>
              <span class="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 uppercase">${c.doc_type || 'qonun'}</span>
            </div>
            <p class="text-xs font-semibold text-slate-200 line-clamp-1">${c.doc_title}</p>
            <p class="text-[11px] text-slate-400 italic line-clamp-2">"${c.quote}"</p>
            <div class="pt-1 flex items-center justify-between text-[11px]">
              <span class="text-slate-500">${c.doc_id} · ${c.edition_date || ''}</span>
              <a href="${c.url}" target="_blank" class="text-sky-400 hover:underline flex items-center gap-1">
                Lex.uz ochish <i class="fa-solid fa-arrow-up-right-from-square text-[9px]"></i>
              </a>
            </div>
          `;
          citList.appendChild(card);
        });
      } else {
        citList.innerHTML = `<p class="text-xs text-slate-500 col-span-2">Ushbu savol uchun tasdiqlovchi manba topilmadi.</p>`;
      }

      // Contexts
      const ctxList = document.getElementById("contextsList");
      ctxList.innerHTML = "";
      document.getElementById("contextsCount").innerText = item.contexts ? item.contexts.length : 0;
      if (item.contexts && item.contexts.length > 0) {
        item.contexts.forEach(ctx => {
          const div = document.createElement("div");
          div.className = "p-3 rounded-xl bg-slate-950/70 border border-slate-800 space-y-2";
          div.innerHTML = `
            <div class="flex items-center justify-between text-xs">
              <span class="font-bold text-slate-300">#${ctx.rank} ${ctx.doc_id} (${ctx.article})</span>
              <div class="space-x-2 text-[10px]">
                <span class="text-slate-400">Rerank: <b class="text-indigo-400">${ctx.rerank_score}</b></span>
                <span class="text-slate-400">Retrieval: <b class="text-sky-400">${ctx.retrieval_score}</b></span>
              </div>
            </div>
            <pre class="text-xs text-slate-400 whitespace-pre-wrap leading-relaxed max-h-36 overflow-y-auto">${ctx.text}</pre>
          `;
          ctxList.appendChild(div);
        });
      }
    }

    function copyResultJson() {
      if (!activeResultData) return;
      navigator.clipboard.writeText(JSON.stringify(activeResultData, null, 2));
      alert("JSON nusxalandi!");
    }

    function loadSampleBatch() {
      const sample = {
        "items": [
          {
            "id": "q001",
            "question": "Yillik asosiy mehnat ta'tilining eng kam muddati qancha?",
            "language": "uz-Latn"
          },
          {
            "id": "q002",
            "question": "Какой срок исковой давности установлен Гражданским кодексом?",
            "language": "ru"
          },
          {
            "id": "q003",
            "question": "O'zbekiston Respublikasi fuqaroligi qanday asoslarda bekor qilinadi?",
            "language": "uz-Latn"
          }
        ]
      };
      document.getElementById("batchJsonArea").value = JSON.stringify(sample.items, null, 2);
    }

    async function executeBatchQuery() {
      const runId = document.getElementById("batchRunId").value.trim();
      const batchId = document.getElementById("batchBatchId").value.trim();
      const bIndex = parseInt(document.getElementById("batchIndex").value, 10);
      const bTotal = parseInt(document.getElementById("batchTotal").value, 10);
      const batchStyleEl = document.getElementById("batchStyle");
      const batchStyle = batchStyleEl ? batchStyleEl.value : "simple";
      const rawItems = document.getElementById("batchJsonArea").value.trim();

      let items = [];
      try {
        items = JSON.parse(rawItems);
        if (!Array.isArray(items)) throw new Error("items massiv bo'lishi kerak");
      } catch (e) {
        return alert("JSON xatosi: " + e.message);
      }

      const submitBtn = document.getElementById("batchSubmitBtn");
      submitBtn.disabled = true;
      submitBtn.innerHTML = `<i class="fa-solid fa-spinner animate-spin"></i> Batch ishlanmoqda...`;

      try {
        const payload = {
          run_id: runId,
          batch_id: batchId,
          batch_index: bIndex,
          total_batches: bTotal,
          options: {
            top_k: 5,
            return_contexts: true,
            return_debug: true,
            temperature: 0.0,
            style: batchStyle
          },
          items: items
        };

        const res = await fetch("/v1/rag/batch", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + getApiKey(),
          },
          body: JSON.stringify(payload)
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || ("HTTP " + res.status));
        }

        const data = await res.json();
        renderBatchResult(data);
      } catch (err) {
        alert("Batch xatosi: " + err.message);
      } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `<i class="fa-solid fa-play"></i> Batchni yuborish`;
      }
    }

    function renderBatchResult(batch) {
      const container = document.getElementById("batchResultContainer");
      container.classList.remove("hidden");

      document.getElementById("batchSummaryTotal").innerText = "Jami: " + batch.summary.total;
      document.getElementById("batchSummaryOk").innerText = "OK: " + batch.summary.ok;
      document.getElementById("batchSummaryNoAns").innerText = "No-Ans: " + batch.summary.no_answer;
      document.getElementById("batchSummaryErr").innerText = "Error: " + batch.summary.error;
      document.getElementById("batchTotalTime").innerText = "Jami vaqt: " + batch.timing.total_ms + " ms";

      const tbody = document.getElementById("batchTableBody");
      tbody.innerHTML = "";

      batch.results.forEach(r => {
        const tr = document.createElement("tr");
        tr.className = "hover:bg-slate-900/50";
        const citCount = r.citations ? r.citations.length : 0;
        const citBadge = citCount > 0 ? `<span class="px-2 py-0.5 rounded bg-sky-500/20 text-sky-400 font-bold">${citCount} ta</span>` : `<span class="text-slate-500">0</span>`;
        tr.innerHTML = `
          <td class="p-3 font-mono font-bold text-sky-400">${r.id}</td>
          <td class="p-3">
            <span class="px-2 py-0.5 rounded text-[10px] font-bold uppercase ${r.status === 'ok' ? 'bg-emerald-500/20 text-emerald-400' : (r.status === 'no_answer' ? 'bg-amber-500/20 text-amber-400' : 'bg-rose-500/20 text-rose-400')}">
              ${r.status}
            </span>
          </td>
          <td class="p-3 max-w-md truncate text-slate-300">${r.answer || '-'}</td>
          <td class="p-3">${citBadge}</td>
          <td class="p-3 text-slate-400">${r.timing ? r.timing.total_ms : 0}</td>
        `;
        tbody.appendChild(tr);
      });
    }

    async function loadHistory() {
      try {
        const res = await fetch("/api/dashboard/recent-batches?limit=20");
        if (!res.ok) return;
        const list = await res.json();
        const tbody = document.getElementById("historyTableBody");
        tbody.innerHTML = "";

        if (list.length === 0) {
          tbody.innerHTML = `<tr><td colspan="7" class="p-4 text-center text-slate-500">Hozircha saqlangan batchlar yo'q.</td></tr>`;
          return;
        }

        list.forEach(b => {
          const sum = b.summary || { total: 0, ok: 0, no_answer: 0, error: 0 };
          const tr = document.createElement("tr");
          tr.className = "hover:bg-slate-900/50";
          tr.innerHTML = `
            <td class="p-3 font-mono text-sky-400 font-bold">${b.batch_id}</td>
            <td class="p-3 text-slate-400">${b.run_id}</td>
            <td class="p-3"><span class="px-2 py-0.5 rounded text-[10px] bg-slate-800 text-slate-300 font-bold uppercase">${b.status}</span></td>
            <td class="p-3 text-slate-300">${sum.total} (${sum.ok} / ${sum.no_answer} / ${sum.error})</td>
            <td class="p-3 text-slate-400">${b.total_ms || '-'}</td>
            <td class="p-3 text-slate-500 text-[11px]">${b.created_at.slice(0, 19).replace('T', ' ')}</td>
            <td class="p-3">
              <button onclick="inspectBatch('${b.batch_id}')" class="text-xs text-sky-400 hover:underline">Ko'rish</button>
            </td>
          `;
          tbody.appendChild(tr);
        });
      } catch (e) {
        console.error("History load error", e);
      }
    }

    async function inspectBatch(batchId) {
      try {
        const res = await fetch("/v1/rag/batch/" + batchId, {
          headers: { "Authorization": "Bearer " + getApiKey() }
        });
        if (!res.ok) throw new Error("HTTP " + res.status);
        const data = await res.json();
        switchTab("batch");
        renderBatchResult(data);
      } catch (e) {
        alert("Batchni yuklab bo'lmadi: " + e.message);
      }
    }
  </script>
</body>
</html>
"""


def get_dashboard_html() -> HTMLResponse:
    """Возвращает скомпилированный HTML интерфейс дашборда."""
    return HTMLResponse(content=DASHBOARD_HTML, status_code=200)
