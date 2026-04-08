// Multi-input wasm benchmark — load once, run several IPA strings.
// Measures wall ms per call, audio length, and RTF.

const fs = require("fs");
const path = require("path");
const os = require("os");

const { WasmFerroModelBuilder } = require("/tmp/ferrocarril_node_pkg/ferrocarril_wasm");

const WEIGHTS_DIR = path.join(os.homedir(), "ferrocarril", "ferrocarril_weights");
const SAMPLE_RATE = 24000;

function hrtimeMs() {
    const [s, ns] = process.hrtime();
    return s * 1000 + ns / 1_000_000;
}

const configJson = fs.readFileSync(path.join(WEIGHTS_DIR, "config.json"), "utf8");
const metadataJson = fs.readFileSync(path.join(WEIGHTS_DIR, "model", "metadata.json"), "utf8");
const voicesJson = fs.readFileSync(path.join(WEIGHTS_DIR, "voices", "voices.json"), "utf8");
const metadata = JSON.parse(metadataJson);
const voices = JSON.parse(voicesJson);

const tensorFiles = [];
for (const [, comp] of Object.entries(metadata.components)) {
    for (const [, tmeta] of Object.entries(comp.parameters)) {
        tensorFiles.push(tmeta.file);
    }
}
const voiceMeta = voices.voices["af_heart"];

const builder = new WasmFerroModelBuilder(configJson, metadataJson, voicesJson);
const loadStart = hrtimeMs();
for (const file of tensorFiles) {
    builder.add_model_blob(file, fs.readFileSync(path.join(WEIGHTS_DIR, "model", file)));
}
const voiceBytes = fs.readFileSync(path.join(WEIGHTS_DIR, "voices", voiceMeta.file));
builder.add_voice_blob(voiceMeta.file, voiceBytes);
console.log(`blob load: ${(hrtimeMs() - loadStart).toFixed(0)} ms`);

const buildStart = hrtimeMs();
const model = builder.build();
console.log(`model build: ${(hrtimeMs() - buildStart).toFixed(0)} ms`);
console.log("");

// IPA inputs of varying length. The first is the existing kmodel
// fixture; the others are progressively longer phoneme strings drawn
// from the Kokoro vocabulary so the model accepts them without
// dropping characters.
const inputs = [
    ["short  (5 ph)", "hɛlqʊ"],
    ["medium (10 ph)", "hɛlqʊ wˈɜɹld"],
    ["medium (20 ph)", "ðə kwˈɪk bɹˈaʊn fˈɑks"],
    ["long  (~50 ph)", "ðə kwˈɪk bɹˈaʊn fˈɑks dʒˈʌmps ˌoʊvɚ ðə lˈeɪzi dˈɔɡ"],
];

console.log("input            ph  audio(s)  wall(s)   RTF");
console.log("--------------------------------------------------");
for (const [label, ipa] of inputs) {
    const t0 = hrtimeMs();
    const samples = model.synthesize_ipa(ipa, voiceBytes, 1.0);
    const wallMs = hrtimeMs() - t0;
    const audioSec = samples.length / SAMPLE_RATE;
    const wallSec = wallMs / 1000;
    const rtf = audioSec / wallSec;
    const phCount = [...ipa].length;
    console.log(
        `${label.padEnd(16)} ${String(phCount).padStart(3)}  ` +
        `${audioSec.toFixed(2).padStart(7)}  ${wallSec.toFixed(2).padStart(7)}  ${rtf.toFixed(3).padStart(5)}`
    );
}

model.free();