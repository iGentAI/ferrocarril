// ferrocarril-wasm browser demo

import init, { WasmFerroModelBuilder } from "./pkg/ferrocarril_wasm.js";

const smokeBtn = document.getElementById("smokeBtn");
const smokeStatus = document.getElementById("smokeStatus");
const loadBtn = document.getElementById("loadBtn");
const speakBtn = document.getElementById("speakBtn");
const ipaInput = document.getElementById("ipaInput");
const fullStatus = document.getElementById("fullStatus");
const player = document.getElementById("player");

const SAMPLE_RATE = 24000;

let loadedModel = null;
let loadedVoicePack = null;
let wasmReady = null;

function statusOk(el, msg) {
    el.textContent = msg;
    el.classList.remove("err");
    el.classList.add("ok");
}
function statusErr(el, msg) {
    el.textContent = msg;
    el.classList.remove("ok");
    el.classList.add("err");
}
function statusNeutral(el, msg) {
    el.textContent = msg;
    el.classList.remove("err");
    el.classList.remove("ok");
}

async function ensureWasm() {
    if (!wasmReady) {
        wasmReady = init();
    }
    return wasmReady;
}

// -----------------------------------------------------------------------------
// Smoke test: just verify the bindings load.
// -----------------------------------------------------------------------------

smokeBtn.addEventListener("click", async () => {
    smokeBtn.disabled = true;
    statusNeutral(smokeStatus, "Loading wasm module...");
    try {
        await ensureWasm();
        statusNeutral(smokeStatus, "Constructing an empty builder...");
        // The builder accepts any string; we give it a minimal
        // metadata skeleton so the constructor itself succeeds. We
        // deliberately do NOT call .build() here, because that would
        // require real weight blobs.
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
        const builder = new WasmFerroModelBuilder(
            minimalConfig,
            minimalMetadata,
            "",
        );
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

// -----------------------------------------------------------------------------
// Full inference: fetch every weight blob, build the model, synthesise.
// -----------------------------------------------------------------------------

async function fetchText(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`fetch ${url} -> HTTP ${r.status}`);
    return r.text();
}

async function fetchBytes(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`fetch ${url} -> HTTP ${r.status}`);
    return new Uint8Array(await r.arrayBuffer());
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

async function fetchAll(files, pathPrefix, onProgress) {
    // Fetch with bounded parallelism to avoid saturating the browser.
    const results = new Map();
    const CONCURRENCY = 8;
    let idx = 0;
    let done = 0;

    async function worker() {
        while (idx < files.length) {
            const myIdx = idx++;
            const file = files[myIdx];
            const url = pathPrefix + file;
            const bytes = await fetchBytes(url);
            results.set(file, bytes);
            done++;
            if (onProgress) onProgress(done, files.length);
        }
    }
    const workers = [];
    for (let i = 0; i < CONCURRENCY; i++) workers.push(worker());
    await Promise.all(workers);
    return results;
}

loadBtn.addEventListener("click", async () => {
    loadBtn.disabled = true;
    speakBtn.disabled = true;
    if (player.src) {
        URL.revokeObjectURL(player.src);
        player.removeAttribute("src");
        player.load();
    }
    if (loadedModel) {
        try { loadedModel.free(); } catch (_) {}
        loadedModel = null;
    }
    loadedVoicePack = null;
    try {
        statusNeutral(fullStatus, "Loading wasm module...");
        await ensureWasm();

        statusNeutral(fullStatus, "Fetching config.json / metadata.json / voices.json...");
        const [configJson, metadataJson, voicesJson] = await Promise.all([
            fetchText("./weights/config.json"),
            fetchText("./weights/model/metadata.json"),
            fetchText("./weights/voices/voices.json"),
        ]);

        const metadata = JSON.parse(metadataJson);
        const tensorFiles = enumerateTensorFiles(metadata);
        statusNeutral(
            fullStatus,
            `Fetching ${tensorFiles.length} weight blobs (0 / ${tensorFiles.length})...`,
        );

        const tensorBlobs = await fetchAll(
            tensorFiles,
            "./weights/model/",
            (done, total) => {
                statusNeutral(
                    fullStatus,
                    `Fetching weight blobs (${done} / ${total})...`,
                );
            },
        );

        statusNeutral(fullStatus, "Fetching voice pack (af_heart)...");
        const voices = JSON.parse(voicesJson);
        const voiceName = "af_heart";
        const voiceMeta = voices.voices[voiceName];
        if (!voiceMeta) throw new Error(`voice '${voiceName}' not in voices.json`);
        const voicePackBytes = await fetchBytes("./weights/" + voiceMeta.file);

        statusNeutral(fullStatus, "Constructing builder and registering blobs...");
        const builder = new WasmFerroModelBuilder(
            configJson,
            metadataJson,
            voicesJson,
        );
        for (const [file, bytes] of tensorBlobs.entries()) {
            builder.add_model_blob(file, bytes);
        }
        builder.add_voice_blob(voiceMeta.file, voicePackBytes);

        statusNeutral(fullStatus, "Building model (wiring tensors into FerroModel)...");
        loadedModel = builder.build();
        loadedVoicePack = voicePackBytes;

        statusOk(
            fullStatus,
            `Model loaded. ${tensorFiles.length} tensor blobs + 1 voice pack registered. Ready to synthesise.`,
        );
        speakBtn.disabled = false;
    } catch (err) {
        statusErr(fullStatus, "FAIL: " + (err && err.message ? err.message : err));
        if (loadedModel) {
            try { loadedModel.free(); } catch (_) {}
            loadedModel = null;
        }
    } finally {
        loadBtn.disabled = false;
    }
});

function floatsToWavBlob(samples, sampleRate) {
    // Encode Float32Array as 16-bit PCM little-endian WAV.
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

speakBtn.addEventListener("click", async () => {
    if (!loadedModel || !loadedVoicePack) return;
    speakBtn.disabled = true;
    try {
        const ipa = (ipaInput.value || "").trim();
        if (!ipa) throw new Error("IPA input is empty");

        statusNeutral(fullStatus, `Synthesising "${ipa}"...`);
        const t0 = performance.now();
        const samples = loadedModel.synthesize_ipa(ipa, loadedVoicePack, 1.0);
        const ms = (performance.now() - t0).toFixed(0);

        const durationSec = samples.length / SAMPLE_RATE;
        const blob = floatsToWavBlob(samples, SAMPLE_RATE);
        if (player.src) URL.revokeObjectURL(player.src);
        player.src = URL.createObjectURL(blob);

        statusOk(
            fullStatus,
            `Done. ${samples.length} samples (${durationSec.toFixed(2)} s @ ${SAMPLE_RATE} Hz) in ${ms} ms.`,
        );
    } catch (err) {
        statusErr(fullStatus, "FAIL: " + (err && err.message ? err.message : err));
    } finally {
        speakBtn.disabled = false;
    }
});

window.addEventListener("DOMContentLoaded", () => {
    smokeBtn.click();
});