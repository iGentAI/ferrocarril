// Benchmark the wasm build of Ferrocarril TTS in Node.js or Bun.
//
// Usage:
//   node bench_wasm.js
//   bun  bench_wasm.js
//
// Requires:
//   * /tmp/ferrocarril_node_pkg/         — wasm-bindgen --target nodejs output
//   * ~/ferrocarril/ferrocarril_weights/ — converted Kokoro weights
//
// Output: wall clock per inference call, audio duration, and real-time
// factor (audio_seconds / inference_seconds). A factor > 1.0 means the
// build is faster than real-time.

const fs = require("fs");
const path = require("path");
const os = require("os");

const {
    WasmFerroModelBuilder,
} = require("/tmp/ferrocarril_node_pkg/ferrocarril_wasm");

const WEIGHTS_DIR = path.join(os.homedir(), "ferrocarril", "ferrocarril_weights");
const SAMPLE_RATE = 24000;

function hrtimeMs() {
    const [s, ns] = process.hrtime();
    return s * 1000 + ns / 1_000_000;
}

function runtimeName() {
    if (typeof Bun !== "undefined") return `Bun ${Bun.version}`;
    if (typeof process !== "undefined" && process.versions) {
        return `Node ${process.versions.node}`;
    }
    return "unknown";
}

function main() {
    console.log(`ferrocarril-wasm benchmark (${runtimeName()})`);
    console.log("====================================");

    if (!fs.existsSync(WEIGHTS_DIR)) {
        console.error(
            `!! weights directory not found at ${WEIGHTS_DIR}\n` +
            `   Run:\n` +
            `     python3 weight_converter.py --huggingface hexgrad/Kokoro-82M --output ferrocarril_weights\n` +
            `   from the repo root first.`,
        );
        process.exit(1);
    }

    // -------------------------------------------------------------------
    // Load metadata JSONs
    // -------------------------------------------------------------------
    const configJson = fs.readFileSync(
        path.join(WEIGHTS_DIR, "config.json"),
        "utf8",
    );
    const metadataJson = fs.readFileSync(
        path.join(WEIGHTS_DIR, "model", "metadata.json"),
        "utf8",
    );
    const voicesJson = fs.readFileSync(
        path.join(WEIGHTS_DIR, "voices", "voices.json"),
        "utf8",
    );
    const metadata = JSON.parse(metadataJson);
    const voices = JSON.parse(voicesJson);

    // -------------------------------------------------------------------
    // Enumerate tensor files + voice blob
    // -------------------------------------------------------------------
    const tensorFiles = [];
    for (const [, comp] of Object.entries(metadata.components)) {
        for (const [, tmeta] of Object.entries(comp.parameters)) {
            tensorFiles.push(tmeta.file);
        }
    }
    const voiceName = "af_heart";
    const voiceMeta = voices.voices[voiceName];
    if (!voiceMeta) {
        throw new Error(`voice '${voiceName}' not in voices.json`);
    }

    console.log(`Model tensors: ${tensorFiles.length}`);
    console.log(`Voice pack:    ${voiceName} (voices/${voiceMeta.file})`);

    // -------------------------------------------------------------------
    // Populate builder
    // -------------------------------------------------------------------
    const builder = new WasmFerroModelBuilder(
        configJson,
        metadataJson,
        voicesJson,
    );

    const loadStart = hrtimeMs();
    let totalBytes = 0;
    for (const file of tensorFiles) {
        const bytes = fs.readFileSync(path.join(WEIGHTS_DIR, "model", file));
        builder.add_model_blob(file, bytes);
        totalBytes += bytes.length;
    }
    // The `file` field in voices.json is relative to the `voices/`
    // subdirectory of the weights pack (e.g. `"af_heart.bin"`), so the
    // actual disk path is WEIGHTS_DIR/voices/<file>.
    const voiceBytes = fs.readFileSync(
        path.join(WEIGHTS_DIR, "voices", voiceMeta.file),
    );
    builder.add_voice_blob(voiceMeta.file, voiceBytes);
    totalBytes += voiceBytes.length;
    const loadEnd = hrtimeMs();
    console.log(
        `Blob load: ${(totalBytes / (1024 * 1024)).toFixed(1)} MB into builder in ${(loadEnd - loadStart).toFixed(0)} ms`,
    );

    // -------------------------------------------------------------------
    // Build the model (wire tensors into FerroModel)
    // -------------------------------------------------------------------
    const buildStart = hrtimeMs();
    const model = builder.build();
    const buildEnd = hrtimeMs();
    console.log(
        `Model build: ${(buildEnd - buildStart).toFixed(0)} ms (deserialises every tensor from blob bytes)`,
    );
    console.log("");

    // -------------------------------------------------------------------
    // Single inference run on the canonical kmodel fixture input.
    // This produces ~1.4 s of audio and is the same input the native
    // end_to_end_real_voice_test uses.
    // -------------------------------------------------------------------
    const ipa = "hɛlqʊ";
    console.log(`Inference: IPA="${ipa}"  (canonical kmodel fixture)`);

    const t0 = hrtimeMs();
    const samples = model.synthesize_ipa(ipa, voiceBytes, 1.0);
    const t1 = hrtimeMs();

    const wallMs = t1 - t0;
    const wallSec = wallMs / 1000;
    const audioSec = samples.length / SAMPLE_RATE;
    const rtx = audioSec / wallSec;
    console.log(
        `  ${samples.length} samples = ${audioSec.toFixed(3)} s audio in ${wallSec.toFixed(2)} s wall -> ${rtx.toFixed(3)}x real-time`,
    );
    if (rtx >= 1.0) {
        console.log("  >> faster than real-time");
    } else {
        const slowdown = 1 / rtx;
        console.log(
            `  >> slower than real-time by a factor of ${slowdown.toFixed(1)}x`,
        );
    }

    model.free();
    console.log("Done.");
}

main();