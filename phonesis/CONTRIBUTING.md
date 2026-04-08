
First off, thank you for considering contributing to Phonesis! It's people like you who make this project a great tool for the Rust community.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report. Following these guidelines helps maintainers and the community understand your report, reproduce the behavior, and find related reports.

**Before Submitting A Bug Report:**

* Check the GitHub issues to see if the problem has already been reported
* Ensure you're running the latest version of the library
* Check if the problem occurs with a minimal set of dependencies

**How to Submit A Good Bug Report:**

* Use a clear and descriptive title
* Describe the exact steps to reproduce the problem
* Describe the behavior you observed and the behavior you expected
* Include code samples and, if possible, a minimal reproducible example
* Include your Rust and Phonesis versions

Please file issues at
[iGentAI/ferrocarril](https://github.com/iGentAI/ferrocarril/issues)
(phonesis lives inside the ferrocarril workspace).

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion, including completely new features and minor improvements to existing functionality.

**How to Submit A Good Enhancement Suggestion:**

* Use a clear and descriptive title
* Provide a step-by-step description of the suggested enhancement
* Explain why this enhancement would be useful to most Phonesis users
* List any alternative solutions or features you've considered

### Your First Code Contribution

Unsure where to begin contributing? Check the
[open issues](https://github.com/iGentAI/ferrocarril/issues) on the
ferrocarril repository — look for labels like `good first issue` or
`help wanted`, or for any issue tagged with `phonesis`.

### Pull Requests

* Fill in the required template
* Title your pull request clearly and concisely
* Include a reference to any relevant issues
* Include before-and-after examples if your change affects output
* Update the README.md with details of major changes
* Make sure your code passes all tests
* Add new tests for new functionality

## Styleguides

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line
* Consider starting the commit message with an applicable emoji:
    * 🎨 `:art:` when improving the format/structure of the code
    * 🐎 `:racehorse:` when improving performance
    * 📝 `:memo:` when writing docs
    * 🐛 `:bug:` when fixing a bug
    * 🔥 `:fire:` when removing code or files

### Rust Styleguide

All Rust code should adhere to the [Rust Style Guide](https://github.com/rust-lang/rfcs/blob/master/text/0430-finalizing-naming-conventions.md) and pass [clippy](https://github.com/rust-lang/rust-clippy) checks.

* Run `cargo fmt` before committing
* Run `cargo clippy` and resolve any warnings
* Add documentation to all public APIs
* Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)

## Development Setup

1. Fork the repository at
   [iGentAI/ferrocarril](https://github.com/iGentAI/ferrocarril)
2. Clone your fork locally
3. Create a branch for your changes
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Make your changes and run tests
   ```bash
   cd phonesis
   cargo test --release
   cargo clippy
   cargo fmt --check
   ```
5. Commit your changes using a descriptive commit message
6. Push your branch to GitHub
7. Submit a pull request

## Project Structure

The Phonesis project is organized as follows:

```
phonesis/
├── src/
│   ├── lib.rs                   # Main library entry point
│   ├── error.rs                 # Error types and handling
│   ├── phoneme.rs               # Phoneme representation
│   ├── ferrocarril_adapter.rs   # Ferrocarril TTS adapter
│   ├── dictionary/              # Dictionary component
│   │   ├── mod.rs               # Dictionary public API
│   │   └── trie.rs              # Compact trie implementation
│   ├── rules/                   # Rule engine
│   ├── normalizer/              # Text normalization
│   └── english/                 # English-specific implementation
├── examples/                    # Example usage
├── tests/                       # Integration tests
├── embedded_dictionary_data.rs  # CMU dictionary compiled in
├── Cargo.toml                   # Package configuration
└── README.md                    # Documentation
```

### Dependencies Policy

Phonesis aims to have zero external dependencies for its core functionality. Additional features may be enabled through feature flags.

Consider the following when adding dependencies:
* Can the functionality be implemented in Rust without dependencies?
* If a dependency is necessary, is it well-maintained and widely used?
* Can the dependency be made optional through feature flags?

## Testing Guidelines

* All code should be covered by tests
* New features should include both unit and integration tests
* Tests should be deterministic and not rely on external resources

## Dictionary and Data Contributions

If you're contributing a new language or dictionary:
1. Place dictionary data in `src/[language]/data/`
2. Include attribution and license information
3. Add documentation on sources and processing
4. Include validation tests for the dictionary
5. Ensure the dictionary is memory-efficient

## Documentation

* All public items must be documented
* Examples should be included where appropriate
* Code samples in documentation should be tested

## Release Process

1. Update version in Cargo.toml
2. Create a new git tag
3. Submit a pull request for review
4. Once approved, merge and publish to crates.io

## Becoming a Maintainer

Contributors who have made substantial and valuable contributions may be given commit-access to the project. Reach out to existing maintainers if you're interested.

## Questions?

Feel free to reach out to the maintainers if you have any questions about contributing.

Thank you for your interest in improving Phonesis!