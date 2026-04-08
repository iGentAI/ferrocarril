// ferrocarril-wasm browser demo runner.
//
// Flow:
//   1. Fetch weights JSON metadata (config, metadata, voices)
//   2. Populate the voice picker
//   3. On "Load model": fetch every tensor + selected voice blob,
//      streaming through an IndexedDB cache so returning visitors
//      skip the ~340 MB cold download, then build the FerroModel.
//   4. On "Synthesise": call WasmFerroModel.synthesize_text with the
//      English text, the selected voice pack bytes, and a speed
//      factor, and play the resulting 24 kHz PCM as a WAV blob.
//
// Weights host configuration is driven by two <meta> tags in
// index.html that the Pages deploy workflow substitutes at build
// time:
//
//   <meta name="ferrocarril-weights-url" content="{{url}}">
//   <meta name="ferrocarril-weights-layout" content="{{nested|flat}}">
//
// The URL is the base prefix under which blobs live. The layout
// decides whether `/` in asset paths is preserved (`nested`, default)
// or rewritten to `__` (`flat`, required for GitHub Release assets
// which cannot contain `/`). A `?weights=<url>` query parameter can
// override the URL for debugging; append `&layout=flat` to also
// override the layout.

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
    return value === "flat" ? "flat" : "nested";
}

/**
 * Given a path relative to the weights root (e.g. "model/metadata.json"
 * or "voices/af_heart.bin"), return the absolute URL to fetch it from,
 * applying the configured layout rewrite if needed. For `nested` the
 * path is passed through verbatim; for `flat` every `/` is replaced
 * with `__` to match the asset naming produced by the release-weights
 * workflow (GitHub Release asset names cannot contain `/`).
 */
function assetUrl(relPath) {
    const base = getWeightsBaseUrl();
    const layout = getWeightsLayout();
    const normalized = relPath.replace(/^\/+/, "");
    const mapped = layout === "flat" ? normalized.replace(/\//g, "__") : normalized;
    return base + mapped;
}

/**
 * IndexedDB cache key for a given relative path. Always keyed by the
 * nested-form path plus the base URL: this way a layout switch for
 * the same host reuses the cached bytes (same content, different URL
 * shape), and a host switch (new base URL, e.g. new release tag)
 * invalidates the entry automatically.
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
    // Fire-and-forget the cache write so it doesn't block model build.
    idbPut(db, key, bytes).catch((err) => {
        console.warn("IndexedDB put failed for", key, err);
    });
    return { bytes, fromCache: false };
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
    // Prefer af_heart as default, fall back to first name.
    const defaultName = names.includes("af_heart") ? "af_heart" : names[0];
    voiceSelect.value = defaultName;
    voiceSelect.disabled = false;
}

// ---------------------------------------------------------------------------
// Bulk blob fetch (concurrency-bounded, IDB-cached)
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

// ---------------------------------------------------------------------------
// Load model
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

        statusNeutral(loadStatus, "Fetching config.json / metadata.json / voices.json...");
        const [configJson, metadataJson, voicesJson] = await Promise.all([
            fetchText("config.json"),
            fetchText("model/metadata.json"),
            fetchText("voices/voices.json"),
        ]);

        const metadata = JSON.parse(metadataJson);
        voicesMeta = JSON.parse(voicesJson);
        populateVoicePicker(voicesMeta.voices);

        // `tensorFiles` are the `file` fields from metadata.json — they
        // are relative to the `model/` directory (e.g. "bert/foo.bin").
        // The native loader expects keys in this form. For HTTP fetching
        // we need the full rel path from the weights root.
        const tensorFiles = enumerateTensorFiles(metadata);
        const tensorPaths = tensorFiles.map((f) => "model/" + f);

        const voiceName = voiceSelect.value;
        const voiceMeta = voicesMeta.voices[voiceName];
        if (!voiceMeta) throw new Error(`voice '${voiceName}' missing from voices.json`);

        statusNeutral(
            loadStatus,
            `Fetching ${tensorFiles.length} tensor blobs + 1 voice pack (${voiceName})...`,
        );

        const db = await openIdb();

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

        // Voice pack
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
            // The native loader's key for a tensor is the metadata.json
            // `file` field, which is the path relative to `model/`. We
            // fetched under "model/<file>", so strip the prefix here.
            const key = relPath.startsWith("model/")
                ? relPath.slice("model/".length)
                : relPath;
            builder.add_model_blob(key, bytes);
        }
        builder.add_voice_blob(voiceMeta.file, voicePackBytes);

        loadedModel = builder.build();
        loadProgress.value = 1;

        speakBtn.disabled = false;
        statusOk(
            loadStatus,
            `Model loaded. ${tensorFiles.length} tensors + voice '${voiceName}' ready. ` +
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
// Voice switching: re-fetch if user picks a different voice
// ---------------------------------------------------------------------------

async function ensureVoicePack(voiceName) {
    if (loadedVoicePacks.has(voiceName)) return loadedVoicePacks.get(voiceName);
    if (!voicesMeta) throw new Error("voices metadata not loaded");
    const voiceMeta = voicesMeta.voices[voiceName];
    if (!voiceMeta) throw new Error(`voice '${voiceName}' missing from voices.json`);
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