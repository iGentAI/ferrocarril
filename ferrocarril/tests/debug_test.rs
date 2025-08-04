#[cfg(test)]
mod debug_test {
    use std::path::Path;
    
    #[test]
    fn test_file_paths() {
        println!("Current dir: {:?}", std::env::current_dir().unwrap());
        println!("Config exists: {}", Path::new("ferrocarril_weights/config.json").exists());
        println!("Weights dir exists: {}", Path::new("ferrocarril_weights").exists());
        println!("Model dir exists: {}", Path::new("ferrocarril_weights/model").exists());
        assert!(Path::new("ferrocarril_weights/config.json").exists());
    }
}
