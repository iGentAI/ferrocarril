<div align="center">
  <h3>A Pure Rust Grapheme-to-Phoneme (G2P) Conversion Library</h3>
  <p>
    <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg" alt="License"></a>
  </p>
</div>

## Overview

Phonesis is a pure-Rust library for converting text (graphemes) to
phonetic representations (phonemes). It provides high-quality
grapheme-to-phoneme conversion for English (with future plans for
additional languages) while maintaining a small dependency footprint,
high performance, and a minimal memory profile.

Phonesis ships as an independent crate that is also the G2P front-end
for the [Ferrocarril](https://github.com/iGentAI/ferrocarril) Kokoro
TTS port. It lives inside the ferrocarril repo as a workspace member
but builds, tests, and releases on its own.

## Features

- **Pure Rust.** One library dependency (`lazy_static`) for the
  ARPABET↔IPA mapping tables; no C FFI, no Python, no build-time
  code generation.
- **Embedded pronunciation dictionary.** ~135,000 entries from the
  CMU Pronouncing Dictionary compiled into the library binary at
  build time — no filesystem access needed at runtime, which makes
  it safe for WebAssembly targets.
- **Multi-standard phoneme output.** ARPABET (the internal
  dictionary format) plus a high-fidelity ARPABET→IPA mapping
  tuned for Kokoro / misaki compatibility.
- **Rule-based fallback** for words not in the dictionary, with a
  final character-by-character fallback so the converter is total —
  it never panics or refuses to produce output.
- **TTS-optimized output** including Kokoro/misaki-compatible IPA
  character streams, function-word destressing, possessive-pronoun
  demotion, and Unicode punctuation handling.
- **Comprehensive text normalization** for numbers, currency,
  symbols, abbreviations, and Unicode punctuation.
- **Context-aware pronunciation hooks** (part-of-speech,
  capitalization, surrounding words) exposed through the
  `GraphemeToPhoneme::convert_with_context` API.

## Installation

Phonesis is available either inside the ferrocarril workspace or as
a standalone path dependency:

```toml
# Cargo.toml
[dependencies]
phonesis = { git = "https://github.com/iGentAI/ferrocarril", package = "phonesis" }
```

Or, if you've cloned ferrocarril locally:

```toml
[dependencies]
phonesis = { path = "../ferrocarril/phonesis" }
```

## Quick Start

```rust
use phonesis::{GraphemeToPhoneme, english::EnglishG2P, PhonemeStandard};

fn main() {
    let g2p = EnglishG2P::new().expect("Failed to initialize G2P");

    // Default ARPABET output
    let phonemes = g2p.convert("Hello, world!").unwrap();
    for phoneme in &phonemes {
        print!("{} ", phoneme);
    }
    println!();

    // IPA output (Kokoro-compatible)
    let ipa = g2p.convert_to_standard("Hello", PhonemeStandard::IPA).unwrap();
    println!("IPA: {}", ipa.join(""));
}
```

## Phoneme Standards

### ARPABET (internal dictionary format)

Phonesis uses ARPABET as its core dictionary format, providing:

- ~135,000 word pronunciations from the CMU Pronouncing Dictionary.
- Stress notation: `EH1` (primary), `EH2` (secondary), `EH0`
  (unstressed).
- Multi-character phoneme symbols: `HH`, `TH`, `SH`, `NG`, etc.

### IPA (TTS output format)

The ARPABET→IPA mapping is tuned for modern TTS systems (especially
Kokoro) and produces Kokoro's "encoded IPA" with two-character
diphthongs and stress-dependent `AH`/`ER`:

- **Vowel mapping**: `AE` → `æ`, `EH` → `ɛ`, `IH` → `ɪ`, `UW` → `u`,
  and stress-dependent `AH` → `ə`/`ʌ`, `ER` → `əɹ`/`ɜɹ`.
- **Consonant mapping**: `HH` → `h`, `TH` → `θ`, `DH` → `ð`, `NG` →
  `ŋ`, with affricates `CH` → `ʧ` and `JH` → `ʤ`.
- **Diphthongs**: `AW` → `W`, `AY` → `I`, `EY` → `A`, `OW` → `O`,
  `OY` → `Y` (Kokoro's single-char uppercase encoding).
- **Stress markers**: `EH1` → `ˈɛ`, `EH2` → `ˌɛ`, `EH0` → `ɛ`.

Additionally, the ferrocarril adapter (see below) applies
misaki-style post-processing: word-boundary markers between tokens,
function-word destressing (articles, auxiliaries, most pronouns,
prepositions, conjunctions, modals lose their stress markers),
possessive-pronoun demotion (primary → secondary stress), and
Unicode punctuation tokenisation.

## Usage Examples

### Basic conversion

```rust
use phonesis::{GraphemeToPhoneme, english::EnglishG2P};

let g2p = EnglishG2P::new().unwrap();
let phonemes = g2p.convert("pronunciation").unwrap();
println!(
    "{}",
    phonemes.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(" "),
);
// -> P R AH0 N AH2 N S IY0 EY1 SH AH0 N
```

### Custom configuration

```rust
use phonesis::{
    GraphemeToPhoneme, english::EnglishG2P,
    G2POptions, FallbackStrategy, PhonemeStandard,
};

let options = G2POptions {
    handle_stress: true,
    default_standard: PhonemeStandard::IPA,
    fallback_strategy: FallbackStrategy::UseRules,
};
let g2p = EnglishG2P::with_options(options).unwrap();
let _ = g2p.convert("uncommon word").unwrap();
```

### Converting between standards

```rust
use phonesis::{
    GraphemeToPhoneme, english::EnglishG2P,
    PhonemeStandard, convert_phonemes,
};

let g2p = EnglishG2P::new().unwrap();
let arpabet = g2p.convert("hello").unwrap();
let ipa = convert_phonemes(&arpabet, PhonemeStandard::IPA);
println!("IPA: {}", ipa.join(" "));
// -> IPA: h ɛ ˈl oʊ
```

## Architecture

Phonesis uses a multi-layer approach:

1. **Text normalization** — converts numbers, symbols, abbreviations
   to words.
2. **Tokenization** — splits text into words, numbers, punctuation.
3. **Dictionary lookup** — looks up each word in the embedded CMU
   dictionary (compact-trie indexed).
4. **Rule-based fallback** — applies letter-to-sound rules for
   unknown words.
5. **Character fallback** — spells out any residual symbols so the
   converter is total.
6. **Phoneme generation** — produces the output in the requested
   standard.

The dictionary data is embedded directly in the library's source
code (`embedded_dictionary_data.rs`), with no external files needed
at runtime. This makes Phonesis completely self-contained and safe
to use in any environment, including WebAssembly.

See `DESIGN.md` for the full design rationale.

## Development

### Building from source

```bash
git clone https://github.com/iGentAI/ferrocarril.git
cd ferrocarril/phonesis
cargo build --release
```

Or, from the ferrocarril workspace root, build everything at once:

```bash
cd ferrocarril
cargo build --workspace --release
```

### Running tests

```bash
# phonesis only
cargo test -p phonesis --release

# whole ferrocarril workspace (includes phonesis + ferrocarril tests)
cargo test --workspace --release --no-fail-fast
```

## Contributing

Contributions are welcome. See `CONTRIBUTING.md` for guidelines.
Please file issues and pull requests against
[iGentAI/ferrocarril](https://github.com/iGentAI/ferrocarril).

## License

Dual-licensed under either of:

- [MIT License](LICENSE-MIT)
- [Apache License 2.0](LICENSE-APACHE)

at your option.

## Acknowledgments

Phonesis draws on prior G2P work from:

- The [CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict)
  for the embedded 135k-entry pronunciation table.
- The [misaki](https://github.com/hexgrad/misaki) G2P library
  (Kokoro's upstream) for the IPA encoding, function-word
  destressing rules, and possessive demotion conventions.
- [espeak-ng](https://github.com/espeak-ng/espeak-ng) for the
  general letter-to-sound approach — a future phonesis release may
  add an espeak-ng-style OOV fallback.

## Integration with Ferrocarril TTS

Phonesis is the G2P front-end for the Ferrocarril Kokoro TTS port.
The `ferrocarril_adapter` module provides a small adapter that
wires up the right default options (IPA output, rule-based
fallback, stress handling) and an LRU cache:

```rust
use phonesis::{
    ferrocarril_adapter::{FerrocarrilG2PAdapter, create_ferrocarril_g2p},
    PhonemeStandard,
};

// Direct adapter creation:
let mut adapter = FerrocarrilG2PAdapter::new_english().unwrap();
adapter.set_standard(PhonemeStandard::IPA);

// Or via the factory:
let mut adapter = create_ferrocarril_g2p("en-us").unwrap();
adapter.set_standard(PhonemeStandard::IPA);

// Convert text to TTS-ready phoneme strings:
let phonemes = adapter.convert_for_tts("hello world").unwrap();
// -> "h ɛ l o w ɹ l d" (space-separated IPA characters)
```

The ferrocarril binary itself uses `PhonesisG2P` in
`ferrocarril-core/src/lib.rs`, which wraps this adapter in a
thread-safe `Arc<Mutex<_>>` and plumbs the result through the Kokoro
phoneme vocabulary. See
[ferrocarril's main README](https://github.com/iGentAI/ferrocarril)
for the full inference pipeline.