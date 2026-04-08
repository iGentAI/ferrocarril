// ferrocarril-wasm browser demo runner.
//
// Flow:
//   1. Fetch weights (either a single `weights.tar` for `tar` layout,
//      or 550+ individual blobs for `nested` / `flat` layouts).
//   2. Populate the voice picker from voices.json.
//   3. Build the FerroModel by feeding every tensor + voice blob
//      into WasmFerroModelBuilder, then call .build().
//   4. On "Synthesise": call WasmFerroModel.synthesize_text with the
//      English text, the selected voice pack bytes, and a speed
//      factor, and play the resulting 24 kHz PCM as a WAV blob.
//
// Weights host configuration is driven by two <meta> tags in
// index.html that the Pages deploy workflow substitutes at build
// time:
//
//   <meta name="ferrocarril-weights-url" content="{{url}}">
//   <meta name="ferrocarril-weights-layout" content="{{nested|flat|tar}}">
//
// Supported layouts:
//   - `nested` (default): each tensor/voice blob at its natural
//     relative path under `{url}`, e.g. `{url}/model/metadata.json`
//     and `{url}/voices/af_heart.bin`. Used by the local serve.py
//     development flow, Hugging Face Hub, Cloudflare R2, S3, and any
//     host that supports directory paths in URLs.
//   - `flat`: same per-file layout but with `/` replaced by `__`,
//     e.g. `{url}/model__metadata.json`. Legacy — kept for any
//     existing mirror that predates the tar mode.
//   - `tar`: single `{url}/weights.tar` asset containing every blob;
//     parsed inline on the client. Used for GitHub Releases, which
//     rate-limit release asset uploads to 500/hour (so uploading 605
//     individual files is not viable) and also benefits from the
//     single-fetch cold-load path.
//
// The URL can be overridden at runtime via `?weights=<url>` and the
// layout via `?layout=nested|flat|tar` query parameters, useful for
// debugging or for testing alternative hosts.

import init, { WasmFerroModelBuilder } from "./pkg/ferrocarril_wasm.js";

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------

const loadBtn = document.getElementById("loadBtn");
const loadStatus = document.getElementById("loadStatus");
const loadStatusShort = document.getElementById("loadStatusShort");
const loadProgress = document.getElementById("loadProgress");

const textInput = document.getElementById("textInput");
const ipaInput = document.getElementById("ipaInput");
const voiceSelect = document.getElementById("voiceSelect");
const speedInput = document.getElementById("speedInput");
const speakBtn = document.getElementById("speakBtn");
const speakStatus = document.getElementById("speakStatus");
const speakStatusShort = document.getElementById("speakStatusShort");
const player = document.getElementById("player");

const smokeBtn = document.getElementById("smokeBtn");
const smokeStatus = document.getElementById("smokeStatus");

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SAMPLE_RATE = 24000;
const FETCH_CONCURRENCY = 12;
const IDB_NAME = "ferrocarril-weights";
const IDB_STORE = "blobs";
const IDB_VERSION = 1;
const TAR_BLOCK_SIZE = 512;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let wasmReady = null;
let loadedModel = null;
let loadedVoicePacks = new Map(); // voice_name -> Uint8Array
let currentVoiceName = null;
let voicesMeta = null; // parsed voices.json

// ---------------------------------------------------------------------------
// Status helpers
// ---------------------------------------------------------------------------

function statusOk(el, msg) {
    el.textContent = msg;
    el.classList.remove("err", "warn");
    el.classList.add("ok");
}
function statusErr(el, msg) {
    el.textContent = msg;
    el.classList.remove("ok", "warn");
    el.classList.add("err");
}
function statusNeutral(el, msg) {
    el.textContent = msg;
    el.classList.remove("err", "warn", "ok");
}

// ---------------------------------------------------------------------------
// Weights configuration
// ---------------------------------------------------------------------------

function getWeightsBaseUrl() {
    const metaTag = document.querySelector('meta[name="ferrocarril-weights-url"]');
    const fromMeta = metaTag ? metaTag.getAttribute("content") : "";
    const fromQuery = new URLSearchParams(window.location.search).get("weights");
    let base = (fromQuery || fromMeta || "./weights/").trim();
    if (base && !base.endsWith("/")) base += "/";
    return base;
}

function getWeightsLayout() {
    const metaTag = document.querySelector('meta[name="ferrocarril-weights-layout"]');
    const fromMeta = metaTag ? metaTag.getAttribute("content") : "";
    const fromQuery = new URLSearchParams(window.location.search).get("layout");
    const value = (fromQuery || fromMeta || "nested").trim().toLowerCase();
    if (value === "tar") return "tar";
    if (value === "flat") return "flat";
    return "nested";
}

/**
 * Given a path relative to the weights root (e.g. "model/metadata.json"
 * or "voices/af_heart.bin"), return the absolute URL to fetch it from,
 * applying the configured layout rewrite if needed. Used by the
 * per-file (nested / flat) layouts only — the tar layout fetches a
 * single `weights.tar` asset and parses it inline.
 */
function assetUrl(relPath) {
    const base = getWeightsBaseUrl();
    const layout = getWeightsLayout();
    const normalized = relPath.replace(/^\/+/, "");
    const mapped = layout === "flat" ? normalized.replace(/\//g, "__") : normalized;
    return base + mapped;
}

/**
 * IndexedDB cache key. Keyed by the base URL plus the nested-form
 * relative path, so the cache invalidates when the host changes and
 * survives a layout swap on the same host.
 */
function cacheKey(relPath) {
    return `${getWeightsBaseUrl()}::${relPath}`;
}

// ---------------------------------------------------------------------------
// Wasm init
// ---------------------------------------------------------------------------

async function ensureWasm() {
    if (!wasmReady) {
        wasmReady = init();
    }
    return wasmReady;
}

// ---------------------------------------------------------------------------
// IndexedDB cache
// ---------------------------------------------------------------------------

function openIdb() {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(IDB_NAME, IDB_VERSION);
        req.onupgradeneeded = () => {
            const db = req.result;
            if (!db.objectStoreNames.contains(IDB_STORE)) {
                db.createObjectStore(IDB_STORE);
            }
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

async function idbGet(db, key) {
    return new Promise((resolve, reject) => {
        const tx = db.transaction(IDB_STORE, "readonly");
        const store = tx.objectStore(IDB_STORE);
        const req = store.get(key);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

async function idbPut(db, key, value) {
    return new Promise((resolve, reject) => {
        const tx = db.transaction(IDB_STORE, "readwrite");
        const store = tx.objectStore(IDB_STORE);
        const req = store.put(value, key);
        req.onsuccess = () => resolve();
        req.onerror = () => reject(req.error);
    });
}

async function fetchBlobCached(db, relPath) {
    const key = cacheKey(relPath);
    const cached = await idbGet(db, key).catch(() => null);
    if (cached instanceof Uint8Array) {
        return { bytes: cached, fromCache: true };
    }
    if (cached instanceof ArrayBuffer) {
        return { bytes: new Uint8Array(cached), fromCache: true };
    }
    const url = assetUrl(relPath);
    const r = await fetch(url);
    if (!r.ok) throw new Error(`fetch ${url} -> HTTP ${r.status}`);
    const bytes = new Uint8Array(await r.arrayBuffer());
    idbPut(db, key, bytes).catch((err) => {
        console.warn("IndexedDB put failed for", key, err);
    });
    return { bytes, fromCache: false };
}

// ---------------------------------------------------------------------------
// Streaming fetch with progress (used by the tar loader)
// ---------------------------------------------------------------------------

async function fetchWithProgress(url, onProgress) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`fetch ${url} -> HTTP ${r.status}`);
    const total = parseInt(r.headers.get("content-length") || "0", 10);

    // Older runtimes without ReadableStream fall through to the
    // simple `arrayBuffer()` path with a single progress update.
    if (!r.body || typeof r.body.getReader !== "function") {
        const buf = await r.arrayBuffer();
        if (onProgress) onProgress(buf.byteLength, buf.byteLength);
        return buf;
    }

    const reader = r.body.getReader();
    const chunks = [];
    let received = 0;
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        if (onProgress) onProgress(received, total);
    }

    const out = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
        out.set(chunk, offset);
        offset += chunk.length;
    }
    return out.buffer;
}

// ---------------------------------------------------------------------------
// Minimal inline USTAR tar reader
// ---------------------------------------------------------------------------
//
// Handles the flavour produced by GNU tar with default options:
//   - 512-byte header per entry (name 0-99, size 124-135 as octal
//     ASCII, typeflag at byte 156).
//   - Optional USTAR name prefix at bytes 345-499 for files whose
//     full path exceeds 100 chars; detected by the "ustar" magic at
//     bytes 257-261.
//   - End of archive is two consecutive 512-byte zero blocks; we
//     stop at the first one.
//   - Content is padded to the next 512-byte boundary.
//
// Returns a `Map<string, Uint8Array>` keyed by the path-relative-to-
// the-tar-root (with any leading "./" stripped), where each value is
// a zero-copy subarray view into the input `buffer`. The caller must
// keep `buffer` alive for as long as the values are in use.

function parseTar(buffer) {
    const u8 = new Uint8Array(buffer);
    const files = new Map();
    const decoder = new TextDecoder("utf-8");
    let offset = 0;

    while (offset + TAR_BLOCK_SIZE <= u8.length) {
        const header = u8.subarray(offset, offset + TAR_BLOCK_SIZE);

        // End-of-archive marker: a zero block.
        let allZero = true;
        for (let i = 0; i < TAR_BLOCK_SIZE; i++) {
            if (header[i] !== 0) {
                allZero = false;
                break;
            }
        }
        if (allZero) break;

        // Name (bytes 0-99, null-terminated).
        let nameLen = 0;
        while (nameLen < 100 && header[nameLen] !== 0) nameLen++;
        let name = decoder.decode(header.subarray(0, nameLen));

        // USTAR name prefix (bytes 345-499), if this is a ustar archive.
        // Magic check: bytes 257-261 == "ustar".
        const isUstar =
            header[257] === 0x75 /* u */ &&
            header[258] === 0x73 /* s */ &&
            header[259] === 0x74 /* t */ &&
            header[260] === 0x61 /* a */ &&
            header[261] === 0x72 /* r */;
        if (isUstar) {
            let prefixLen = 0;
            while (prefixLen < 155 && header[345 + prefixLen] !== 0) prefixLen++;
            if (prefixLen > 0) {
                const prefix = decoder.decode(header.subarray(345, 345 + prefixLen));
                name = prefix + "/" + name;
            }
        }

        // Size (bytes 124-135, octal ASCII with possible trailing
        // space or null).
        let sizeStr = "";
        for (let i = 124; i < 136; i++) {
            const c = header[i];
            if (c >= 0x30 /* '0' */ && c <= 0x37 /* '7' */) {
                sizeStr += String.fromCharCode(c);
            }
        }
        const size = sizeStr ? parseInt(sizeStr, 8) : 0;

        // Type flag (byte 156): '0' (0x30) or '\0' (0x00) = regular
        // file; '5' (0x35) = directory; 'x' (0x78) = pax extended
        // header (skip); others (symlinks, etc.) also skipped.
        const typeFlag = header[156];

        offset += TAR_BLOCK_SIZE;

        if ((typeFlag === 0x30 || typeFlag === 0x00) && size > 0 && name) {
            const cleanName = name.startsWith("./") ? name.slice(2) : name;
            if (cleanName) {
                files.set(cleanName, u8.subarray(offset, offset + size));
            }
        }

        // Advance past the content, padded to the next 512-byte
        // boundary. `Math.ceil(size / BLOCK) * BLOCK` handles the
        // size === 0 case correctly (adds 0).
        offset += Math.ceil(size / TAR_BLOCK_SIZE) * TAR_BLOCK_SIZE;
    }

    return files;
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

async function fetchText(relPath) {
    const url = assetUrl(relPath);
    const r = await fetch(url);
    if (!r.ok) throw new Error(`fetch ${url} -> HTTP ${r.status}`);
    return r.text();
}

function enumerateTensorFiles(metadata) {
    const files = [];
    for (const [, comp] of Object.entries(metadata.components)) {
        for (const [, tmeta] of Object.entries(comp.parameters)) {
            files.push(tmeta.file);
        }
    }
    return files;
}

// ---------------------------------------------------------------------------
// Smoke test
// ---------------------------------------------------------------------------

smokeBtn.addEventListener("click", async () => {
    smokeBtn.disabled = true;
    statusNeutral(smokeStatus, "Loading wasm module...");
    try {
        await ensureWasm();
        statusNeutral(smokeStatus, "Constructing an empty builder...");
        const minimalMetadata = JSON.stringify({
            format_version: "1",
            original_file: "(smoke test)",
            components: {},
        });
        const minimalConfig = JSON.stringify({
            n_token: 178,
            hidden_dim: 512,
            n_layer: 3,
            style_dim: 128,
            n_mels: 80,
            max_dur: 50,
            dropout: 0.2,
            text_encoder_kernel_size: 5,
            istftnet: {
                upsample_rates: [10, 6],
                upsample_initial_channel: 512,
                resblock_kernel_sizes: [3, 7, 11],
                resblock_dilation_sizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                upsample_kernel_sizes: [20, 12],
                gen_istft_n_fft: 20,
                gen_istft_hop_size: 5,
            },
            plbert: {
                hidden_size: 768,
                num_attention_heads: 12,
                num_hidden_layers: 12,
                intermediate_size: 2048,
            },
            vocab: {},
        });
        const builder = new WasmFerroModelBuilder(minimalConfig, minimalMetadata, "");
        builder.free();
        statusOk(
            smokeStatus,
            "OK. ferrocarril-wasm loaded and WasmFerroModelBuilder constructed successfully.",
        );
    } catch (err) {
        statusErr(smokeStatus, "FAIL: " + (err && err.message ? err.message : err));
    } finally {
        smokeBtn.disabled = false;
    }
});

// ---------------------------------------------------------------------------
// Voice picker population
// ---------------------------------------------------------------------------

function populateVoicePicker(voices) {
    const names = Object.keys(voices).sort();
    voiceSelect.innerHTML = "";
    for (const name of names) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        voiceSelect.appendChild(opt);
    }
    const defaultName = names.includes("af_heart") ? "af_heart" : names[0];
    voiceSelect.value = defaultName;
    voiceSelect.disabled = false;
}

// ---------------------------------------------------------------------------
// Per-file blob fetch (nested + flat layouts)
// ---------------------------------------------------------------------------
//
// Takes a list of full relative paths from the weights root
// (e.g. "model/bert/foo.bin" or "voices/af_heart.bin") and returns a
// Map keyed by the same relPath so the caller can look up bytes
// deterministically. The layout rewrite happens inside
// fetchBlobCached → assetUrl, so the caller never has to think about
// `nested` vs `flat`.

async function fetchAllBlobs(db, relPaths, onProgress) {
    const results = new Map();
    let done = 0;
    let fromCacheCount = 0;
    let idx = 0;
    async function worker() {
        while (idx < relPaths.length) {
            const myIdx = idx++;
            const relPath = relPaths[myIdx];
            const { bytes, fromCache } = await fetchBlobCached(db, relPath);
            results.set(relPath, bytes);
            done++;
            if (fromCache) fromCacheCount++;
            if (onProgress) onProgress(done, relPaths.length, fromCacheCount);
        }
    }
    const workers = [];
    for (let i = 0; i < FETCH_CONCURRENCY; i++) workers.push(worker());
    await Promise.all(workers);
    return results;
}

async function loadModelFromFiles(db) {
    statusNeutral(loadStatus, "Fetching config.json / metadata.json / voices.json...");
    const [configJson, metadataJson, voicesJson] = await Promise.all([
        fetchText("config.json"),
        fetchText("model/metadata.json"),
        fetchText("voices/voices.json"),
    ]);

    const metadata = JSON.parse(metadataJson);
    voicesMeta = JSON.parse(voicesJson);
    populateVoicePicker(voicesMeta.voices);

    const tensorFiles = enumerateTensorFiles(metadata);
    const tensorPaths = tensorFiles.map((f) => "model/" + f);

    const voiceName = voiceSelect.value;
    const voiceMeta = voicesMeta.voices[voiceName];
    if (!voiceMeta) throw new Error(`voice '${voiceName}' missing from voices.json`);

    statusNeutral(
        loadStatus,
        `Fetching ${tensorFiles.length} tensor blobs + 1 voice pack (${voiceName})...`,
    );

    const tensorBlobsByPath = await fetchAllBlobs(
        db,
        tensorPaths,
        (done, total, fromCache) => {
            loadProgress.value = (done / total) * 0.95;
            loadStatusShort.textContent = `tensors ${done}/${total} (${fromCache} cached)`;
            statusNeutral(
                loadStatus,
                `Fetching tensors ${done}/${total} (${fromCache} served from IndexedDB cache)`,
            );
        },
    );

    statusNeutral(loadStatus, `Fetching voice pack ${voiceName}...`);
    const voiceFetchPath = `voices/${voiceMeta.file}`;
    const { bytes: voicePackBytes, fromCache: voiceFromCache } =
        await fetchBlobCached(db, voiceFetchPath);
    loadedVoicePacks.set(voiceName, voicePackBytes);
    currentVoiceName = voiceName;

    loadProgress.value = 0.98;
    statusNeutral(
        loadStatus,
        `Constructing FerroModel (${tensorFiles.length} tensors, voice ${voiceName}${voiceFromCache ? " [cached]" : ""})...`,
    );

    const builder = new WasmFerroModelBuilder(configJson, metadataJson, voicesJson);
    for (const [relPath, bytes] of tensorBlobsByPath.entries()) {
        const key = relPath.startsWith("model/")
            ? relPath.slice("model/".length)
            : relPath;
        builder.add_model_blob(key, bytes);
    }
    builder.add_voice_blob(voiceMeta.file, voicePackBytes);

    loadedModel = builder.build();
    return { tensorCount: tensorFiles.length, voiceName };
}

// ---------------------------------------------------------------------------
// Tar blob fetch + inline parse (tar layout)
// ---------------------------------------------------------------------------

async function loadModelFromTar(db) {
    const baseUrl = getWeightsBaseUrl();
    const tarCacheKey = `${baseUrl}::weights.tar`;

    // Try IndexedDB cache first.
    statusNeutral(loadStatus, "Checking IndexedDB cache for weights.tar...");
    let tarBuffer = null;
    let tarFromCache = false;
    const cached = await idbGet(db, tarCacheKey).catch(() => null);
    if (cached instanceof Uint8Array) {
        tarBuffer = cached.buffer.slice(cached.byteOffset, cached.byteOffset + cached.byteLength);
        tarFromCache = true;
    } else if (cached instanceof ArrayBuffer) {
        tarBuffer = cached;
        tarFromCache = true;
    }

    if (!tarFromCache) {
        const tarUrl = baseUrl + "weights.tar";
        statusNeutral(loadStatus, `Fetching weights.tar from ${tarUrl}...`);
        tarBuffer = await fetchWithProgress(tarUrl, (done, total) => {
            const mb = (done / 1024 / 1024).toFixed(0);
            const totalMb = total > 0 ? (total / 1024 / 1024).toFixed(0) : "?";
            const pct = total > 0 ? done / total : 0;
            loadProgress.value = pct * 0.75;
            loadStatusShort.textContent = `${mb} / ${totalMb} MB`;
            statusNeutral(
                loadStatus,
                `Fetching weights.tar: ${mb} / ${totalMb} MB`,
            );
        });

        // Cache the tar for returning visitors. Fire-and-forget so
        // failures don't block model construction.
        idbPut(db, tarCacheKey, new Uint8Array(tarBuffer)).catch((err) => {
            console.warn("IndexedDB put failed for weights.tar", err);
        });
    } else {
        const mb = (tarBuffer.byteLength / 1024 / 1024).toFixed(0);
        statusNeutral(
            loadStatus,
            `weights.tar served from IndexedDB cache (${mb} MB)`,
        );
        loadProgress.value = 0.75;
    }

    statusNeutral(loadStatus, "Parsing tar archive...");
    loadProgress.value = 0.78;
    const tarFiles = parseTar(tarBuffer);
    loadStatusShort.textContent = `${tarFiles.size} entries`;

    // Extract JSON metadata from the tar.
    const configBytes = tarFiles.get("config.json");
    const metadataBytes = tarFiles.get("model/metadata.json");
    const voicesBytes = tarFiles.get("voices/voices.json");
    if (!configBytes) throw new Error("weights.tar missing config.json");
    if (!metadataBytes) throw new Error("weights.tar missing model/metadata.json");
    if (!voicesBytes) throw new Error("weights.tar missing voices/voices.json");

    const textDecoder = new TextDecoder("utf-8");
    const configJson = textDecoder.decode(configBytes);
    const metadataJson = textDecoder.decode(metadataBytes);
    const voicesJson = textDecoder.decode(voicesBytes);

    const metadata = JSON.parse(metadataJson);
    voicesMeta = JSON.parse(voicesJson);
    populateVoicePicker(voicesMeta.voices);

    const voiceName = voiceSelect.value;
    const voiceMeta = voicesMeta.voices[voiceName];
    if (!voiceMeta) throw new Error(`voice '${voiceName}' missing from voices.json`);

    // Pre-load every voice pack into `loadedVoicePacks` so subsequent
    // voice switches don't need to re-fetch the tar. Each voice pack
    // is ~522 KB; all 54 together are ~28 MB, which is cheap.
    const voiceNames = Object.keys(voicesMeta.voices);
    for (const vname of voiceNames) {
        const vmeta = voicesMeta.voices[vname];
        const tarKey = `voices/${vmeta.file}`;
        const vbytes = tarFiles.get(tarKey);
        if (vbytes) {
            loadedVoicePacks.set(vname, new Uint8Array(vbytes));
        }
    }
    currentVoiceName = voiceName;
    const selectedVoicePack = loadedVoicePacks.get(voiceName);
    if (!selectedVoicePack) {
        throw new Error(`weights.tar missing voices/${voiceMeta.file}`);
    }

    loadProgress.value = 0.85;
    const tensorFiles = enumerateTensorFiles(metadata);
    statusNeutral(
        loadStatus,
        `Constructing FerroModel (${tensorFiles.length} tensors from tar${tarFromCache ? " [cached]" : ""})...`,
    );

    const builder = new WasmFerroModelBuilder(configJson, metadataJson, voicesJson);
    for (const tensorFile of tensorFiles) {
        const tarKey = "model/" + tensorFile;
        const bytes = tarFiles.get(tarKey);
        if (!bytes) {
            throw new Error(`weights.tar missing ${tarKey}`);
        }
        builder.add_model_blob(tensorFile, bytes);
    }
    builder.add_voice_blob(voiceMeta.file, selectedVoicePack);

    loadedModel = builder.build();
    loadProgress.value = 1;

    return { tensorCount: tensorFiles.length, voiceName };
}

// ---------------------------------------------------------------------------
// Load model button handler
// ---------------------------------------------------------------------------

loadBtn.addEventListener("click", async () => {
    loadBtn.disabled = true;
    speakBtn.disabled = true;
    if (loadedModel) {
        try { loadedModel.free(); } catch (_) {}
        loadedModel = null;
    }
    loadedVoicePacks = new Map();
    currentVoiceName = null;

    const baseUrl = getWeightsBaseUrl();
    const layout = getWeightsLayout();
    try {
        statusNeutral(
            loadStatus,
            `Loading wasm module (weights: ${baseUrl}, layout=${layout})...`,
        );
        loadStatusShort.textContent = "";
        loadProgress.value = 0;
        await ensureWasm();

        const db = await openIdb();

        let result;
        if (layout === "tar") {
            result = await loadModelFromTar(db);
        } else {
            result = await loadModelFromFiles(db);
        }

        loadProgress.value = 1;
        speakBtn.disabled = false;
        statusOk(
            loadStatus,
            `Model loaded. ${result.tensorCount} tensors + voice '${result.voiceName}' ready. ` +
            `Click "Synthesise" to run inference.`,
        );
        loadStatusShort.textContent = "ready";
    } catch (err) {
        statusErr(loadStatus, "FAIL: " + (err && err.message ? err.message : err));
        console.error(err);
        loadStatusShort.textContent = "failed";
        if (loadedModel) {
            try { loadedModel.free(); } catch (_) {}
            loadedModel = null;
        }
    } finally {
        loadBtn.disabled = false;
    }
});

// ---------------------------------------------------------------------------
// Voice switching
// ---------------------------------------------------------------------------

async function ensureVoicePack(voiceName) {
    if (loadedVoicePacks.has(voiceName)) return loadedVoicePacks.get(voiceName);
    if (!voicesMeta) throw new Error("voices metadata not loaded");
    const voiceMeta = voicesMeta.voices[voiceName];
    if (!voiceMeta) throw new Error(`voice '${voiceName}' missing from voices.json`);

    const layout = getWeightsLayout();
    if (layout === "tar") {
        // Tar mode pre-loads every voice during the initial build, so
        // if we got here it's because the voice is genuinely missing
        // from the tar.
        throw new Error(
            `voice '${voiceName}' was not in the loaded weights.tar; ` +
            `re-load the model to pick up new voices`,
        );
    }

    const db = await openIdb();
    const voiceFetchPath = `voices/${voiceMeta.file}`;
    const { bytes } = await fetchBlobCached(db, voiceFetchPath);
    loadedVoicePacks.set(voiceName, bytes);
    return bytes;
}

// ---------------------------------------------------------------------------
// WAV encoding (Float32Array -> 16-bit PCM WAV Blob)
// ---------------------------------------------------------------------------

function floatsToWavBlob(samples, sampleRate) {
    const pcm = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
        let s = Math.max(-1, Math.min(1, samples[i]));
        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }

    const dataSize = pcm.length * 2;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    function writeString(offset, str) {
        for (let i = 0; i < str.length; i++) {
            view.setUint8(offset + i, str.charCodeAt(i));
        }
    }

    writeString(0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, "data");
    view.setUint32(40, dataSize, true);
    new Int16Array(buffer, 44).set(pcm);
    return new Blob([buffer], { type: "audio/wav" });
}

// ---------------------------------------------------------------------------
// Synthesise
// ---------------------------------------------------------------------------

speakBtn.addEventListener("click", async () => {
    if (!loadedModel) return;
    speakBtn.disabled = true;
    loadBtn.disabled = true;
    speakStatusShort.textContent = "";
    if (player.src) {
        URL.revokeObjectURL(player.src);
        player.removeAttribute("src");
        player.load();
    }

    try {
        const voiceName = voiceSelect.value || currentVoiceName;
        const voicePack = await ensureVoicePack(voiceName);

        const text = (textInput.value || "").trim();
        const ipa = (ipaInput.value || "").trim();
        const speedRaw = parseFloat(speedInput.value || "1.0");
        const speed = Number.isFinite(speedRaw) && speedRaw > 0 ? speedRaw : 1.0;

        if (!text && !ipa) {
            throw new Error("please enter some text (or advanced IPA)");
        }

        const label = ipa ? `IPA "${ipa}"` : `"${text}"`;
        statusNeutral(
            speakStatus,
            `Synthesising ${label} with voice ${voiceName} (speed ${speed.toFixed(2)})...`,
        );
        speakStatusShort.textContent = "running…";

        // Yield once so the status update paints before the long
        // synchronous wasm call blocks the main thread.
        await new Promise((r) => requestAnimationFrame(() => r()));

        const t0 = performance.now();
        let samples;
        if (ipa) {
            samples = loadedModel.synthesize_ipa(ipa, voicePack, speed);
        } else {
            samples = loadedModel.synthesize_text(text, voicePack, speed);
        }
        const wallMs = performance.now() - t0;

        const durationSec = samples.length / SAMPLE_RATE;
        const rtx = durationSec / (wallMs / 1000);
        const blob = floatsToWavBlob(samples, SAMPLE_RATE);
        if (player.src) URL.revokeObjectURL(player.src);
        player.src = URL.createObjectURL(blob);

        statusOk(
            speakStatus,
            `Done. ${samples.length} samples (${durationSec.toFixed(2)} s audio @ ${SAMPLE_RATE} Hz) ` +
            `in ${(wallMs / 1000).toFixed(2)} s wall (${rtx.toFixed(2)}× real-time, voice ${voiceName}).`,
        );
        speakStatusShort.textContent = `${(wallMs / 1000).toFixed(1)}s`;
    } catch (err) {
        statusErr(speakStatus, "FAIL: " + (err && err.message ? err.message : err));
        console.error(err);
        speakStatusShort.textContent = "failed";
    } finally {
        speakBtn.disabled = false;
        loadBtn.disabled = false;
    }
});

// ---------------------------------------------------------------------------
// Page init
// ---------------------------------------------------------------------------

window.addEventListener("DOMContentLoaded", () => {
    smokeBtn.click();
});