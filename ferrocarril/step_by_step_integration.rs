//! Step-by-step integration test - start simple and build up
use ferrocarril_core::PhonesisG2P;
use std::error::Error;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_step_1_foundation_g2p() -> Result<(), Box<dyn Error>> {
        println!("🔄 STEP 1: Foundation Layer - Phonesis G2P");
        
        let mut g2p = PhonesisG2P::new("en-us")?;
        let test_text = "Hello world this is a test";
        let phoneme_string = g2p.convert(test_text)?;
        
        println!("  Input text: '{}'", test_text);
        println!("  G2P output: '{}'", phoneme_string);
        
        assert!(!phoneme_string.is_empty(), "G2P should produce phonemes");
        assert!(!phoneme_string.chars().all(|c| c.is_whitespace()), "G2P should produce non-whitespace phonemes");
        
        println!("  ✅ Step 1: Text → G2P → Phonemes SUCCESSFUL");
        
        // Convert to token IDs using simple mapping
        let phoneme_chars: Vec<char> = phoneme_string.chars().collect();
        let mut token_ids = vec![0i64]; // <bos>
        
        for ch in phoneme_chars {
            if ch.is_alphanumeric() || ch == ' ' {
                let id = ch as u8 as i64;
                token_ids.push(id);
            }
        }
        
        token_ids.push(0); // <eos>
        
        println!("  Token IDs: {:?} (length: {})", token_ids, token_ids.len());
        
        println!("  ✅ Step 1 COMPLETE: G2P → Token conversion working");
        
        Ok(())
    }
    
    #[test] 
    fn test_step_2_detect_weight_loading_issue() {
        println!("🔍 STEP 2: Weight Loading Path Detection");
        
        let possible_paths = vec![
            "../real_kokoro_weights",
            "real_kokoro_weights", 
            "../../real_kokoro_weights",
            "/home/sandbox/ferrocarril/real_kokoro_weights"
        ];
        
        for path in &possible_paths {
            let metadata_path = format!("{}/metadata.json", path);
            println!("  Testing path: {}", metadata_path);
            
            if std::path::Path::new(&metadata_path).exists() {
                println!("    ✅ FOUND: {}", metadata_path);
                
                // Try to read
                match std::fs::read_to_string(&metadata_path) {
                    Ok(content) => {
                        println!("    ✅ READABLE: {} bytes", content.len());
                        return;
                    }
                    Err(e) => {
                        println!("    ❌ READ error: {}", e);
                    }
                }
            } else {
                println!("    ❌ not found: {}", metadata_path); 
            }
        }
        
        println!("  ❌ CRITICAL: No metadata file found at any location");
        panic!("Cannot proceed without real weights");
    }
}
