//! Shared helpers for integration-test golden comparisons.
//!
//! Provides a minimal `.npy` loader so Rust tests can diff Rust outputs
//! directly against Python fixtures dumped by
//! `scripts/validate_kmodel.py` / `scripts/validate_bert.py` / etc.
//!
//! This file lives at `tests/common/mod.rs` (not `tests/common.rs`) so
//! Cargo does not compile it as its own test binary.

#![allow(dead_code)]

pub mod npy {
    use std::path::Path;

    /// Tagged numpy array with shape and data.
    pub struct NpyArray {
        pub shape: Vec<usize>,
        pub data: NpyData,
    }

    pub enum NpyData {
        F32(Vec<f32>),
        I64(Vec<i64>),
    }

    impl NpyArray {
        pub fn as_f32(&self) -> &[f32] {
            match &self.data {
                NpyData::F32(v) => v,
                NpyData::I64(_) => panic!("expected f32 numpy data, got i64"),
            }
        }

        pub fn as_i64(&self) -> &[i64] {
            match &self.data {
                NpyData::I64(v) => v,
                NpyData::F32(_) => panic!("expected i64 numpy data, got f32"),
            }
        }

        pub fn num_elements(&self) -> usize {
            self.shape.iter().product()
        }
    }

    /// Load a `.npy` file produced by `numpy.save`.
    ///
    /// Supports versions 1.0, 2.0, 3.0 and dtypes `<f4` (f32 little-endian)
    /// and `<i8` (i64 little-endian). C-contiguous only.
    pub fn load<P: AsRef<Path>>(path: P) -> NpyArray {
        let path_ref = path.as_ref();
        let bytes = std::fs::read(path_ref)
            .unwrap_or_else(|e| panic!("Failed to read {:?}: {}", path_ref, e));

        // Minimum file has 6-byte magic + 2 version bytes + 2-byte v1 header
        // length = 10 bytes. v2/v3 needs 4 more. We re-check below.
        assert!(
            bytes.len() >= 10,
            "file {:?} too short to be a .npy header ({} bytes)",
            path_ref,
            bytes.len()
        );
        assert_eq!(
            &bytes[..6],
            b"\x93NUMPY",
            "not a .npy file: {:?}",
            path_ref
        );

        let major = bytes[6];
        let minor = bytes[7];

        let (header_len, header_start) = match (major, minor) {
            (1, 0) => {
                let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
                (len, 10)
            }
            (2, 0) | (3, 0) => {
                assert!(
                    bytes.len() >= 12,
                    "file {:?} too short for v{}.{} header length field ({} bytes)",
                    path_ref, major, minor, bytes.len()
                );
                let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
                (len, 12)
            }
            (m, n) => panic!("unsupported .npy version {}.{} in {:?}", m, n, path_ref),
        };

        assert!(
            bytes.len() >= header_start + header_len,
            "file {:?} too short for declared header (file={} bytes, needs={}+{})",
            path_ref,
            bytes.len(),
            header_start,
            header_len
        );

        let header = std::str::from_utf8(&bytes[header_start..header_start + header_len])
            .expect("non-utf8 .npy header");

        let shape = parse_shape(header);
        let dtype = parse_dtype(header);
        let fortran = parse_fortran_order(header);
        assert!(
            !fortran,
            "fortran order not supported in {:?} (header: {})",
            path_ref, header
        );

        let data_bytes = &bytes[header_start + header_len..];
        let num_elements: usize = shape.iter().product();

        let data = match dtype.as_str() {
            "<f4" => {
                assert_eq!(
                    data_bytes.len(),
                    num_elements * 4,
                    "f32 payload size mismatch in {:?} (got {} bytes, want {}*4={})",
                    path_ref,
                    data_bytes.len(),
                    num_elements,
                    num_elements * 4
                );
                let mut v = Vec::with_capacity(num_elements);
                for i in 0..num_elements {
                    let b = &data_bytes[i * 4..i * 4 + 4];
                    v.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                }
                NpyData::F32(v)
            }
            "<i8" => {
                assert_eq!(
                    data_bytes.len(),
                    num_elements * 8,
                    "i64 payload size mismatch in {:?} (got {} bytes, want {}*8={})",
                    path_ref,
                    data_bytes.len(),
                    num_elements,
                    num_elements * 8
                );
                let mut v = Vec::with_capacity(num_elements);
                for i in 0..num_elements {
                    let b = &data_bytes[i * 8..i * 8 + 8];
                    v.push(i64::from_le_bytes([
                        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                    ]));
                }
                NpyData::I64(v)
            }
            other => panic!(
                "unsupported dtype `{}` in {:?}; only <f4 and <i8 are implemented",
                other, path_ref
            ),
        };

        NpyArray { shape, data }
    }

    fn parse_shape(header: &str) -> Vec<usize> {
        let key = "'shape':";
        let start = header
            .find(key)
            .unwrap_or_else(|| panic!("no 'shape' key in npy header: {}", header));
        let after = &header[start + key.len()..];
        let lp = after
            .find('(')
            .unwrap_or_else(|| panic!("no '(' in shape: {}", after));
        let rp = after
            .find(')')
            .unwrap_or_else(|| panic!("no ')' in shape: {}", after));
        let inner = &after[lp + 1..rp];
        inner
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse::<usize>()
                    .unwrap_or_else(|_| panic!("bad shape component '{}'", s))
            })
            .collect()
    }

    fn parse_dtype(header: &str) -> String {
        let key = "'descr':";
        let start = header
            .find(key)
            .unwrap_or_else(|| panic!("no 'descr' in npy header: {}", header));
        let after = &header[start + key.len()..];
        let lq = after
            .find('\'')
            .unwrap_or_else(|| panic!("no opening quote on descr: {}", after));
        let after_lq = &after[lq + 1..];
        let rq = after_lq
            .find('\'')
            .unwrap_or_else(|| panic!("no closing quote on descr: {}", after_lq));
        after_lq[..rq].to_string()
    }

    fn parse_fortran_order(header: &str) -> bool {
        let key = "'fortran_order':";
        let start = header
            .find(key)
            .unwrap_or_else(|| panic!("no fortran_order in npy header: {}", header));
        let after = &header[start + key.len()..];
        after.trim_start().starts_with("True")
    }
}

/// Locate the canonical converted Kokoro weights directory relative to the
/// cargo test working directory. Sandbox paths change between runs so we
/// never hard-code absolute paths here.
pub fn find_weights_path() -> Option<String> {
    for candidate in [
        "../ferrocarril_weights",
        "ferrocarril_weights",
        "../../ferrocarril_weights",
    ] {
        if std::path::Path::new(candidate).exists() {
            return Some(candidate.to_string());
        }
    }
    None
}

/// Locate the kmodel fixture directory relative to the cargo test working
/// directory. These are the `.npy` tensors dumped by
/// `scripts/validate_kmodel.py`.
pub fn find_kmodel_fixtures_path() -> Option<String> {
    for candidate in [
        "../tests/fixtures/kmodel",
        "tests/fixtures/kmodel",
        "../../tests/fixtures/kmodel",
    ] {
        if std::path::Path::new(candidate).exists() {
            return Some(candidate.to_string());
        }
    }
    None
}

/// Compute the max absolute elementwise difference between two `f32` slices.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    let mut m = 0.0f32;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > m {
            m = d;
        }
    }
    m
}

/// Compute the mean absolute value of a slice (for reporting only).
pub fn mean_abs(a: &[f32]) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    let s: f32 = a.iter().map(|x| x.abs()).sum();
    s / a.len() as f32
}