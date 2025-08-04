use ferrocarril_core::weights_binary::BinaryWeightLoader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 DIRECT RUST WEIGHT LOADING TEST");
    
    let loader = BinaryWeightLoader::from_directory("../ferrocarril_weights")?;
    
    // Test loading a known weight
    let bert_embeddings = loader.load_component_parameter("bert", "module.embeddings.word_embeddings.weight")?;
    
    println!("✅ RUST WEIGHT LOADING SUCCESS:");
    println!("  Shape: {:?}", bert_embeddings.shape());
    println!("  Data samples: {:?}", &bert_embeddings.data()[0..5]);
    
    println!("\n🎯 RUST BINART WEIGHT LOADING: WORKING");
    
    Ok(())
}
