// Test that we can load real Kokoro weights with our compiled system
use ferrocarril_core::weights_binary::BinaryWeightLoader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 TESTING REAL WEIGHT LOADING WITH COMPILED SYSTEM");
    
    // Load real converted weights
    let loader = BinaryWeightLoader::from_directory("../ferrocarril_weights")?;
    println!("✅ Weight loader created successfully");
    
    // Test loading key components
    let components = loader.list_components();
    println!("📋 Available components: {:?}", components);
    
    // Test loading a few critical weights
    let test_weights = [
        ("bert", "module.embeddings.word_embeddings.weight"),
        ("text_encoder", "module.embedding.weight"),
        ("predictor", "module.lstm.weight_ih_l0"),
    ];
    
    for (component, param) in &test_weights {
        match loader.load_component_parameter(component, param) {
            Ok(tensor) => {
                println!("✅ {}.{}: shape {:?}", component, param, tensor.shape());
            }
            Err(e) => {
                println!("❌ {}.{}: {}", component, param, e);
                return Err(e.into());
            }
        }
    }
    
    println!("\n🎉 REAL WEIGHT LOADING TEST SUCCESSFUL");
    println!("Foundation ready for layer-by-layer validation!");
    
    Ok(())
}
