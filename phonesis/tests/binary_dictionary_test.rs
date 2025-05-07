// Remove the unused import
use phonesis::english::get_default_dictionary;

#[test]
fn test_embedded_dictionary() {
    // Get the default dictionary (now using the embedded data)
    let dict = get_default_dictionary().unwrap();
    
    // Check basic properties
    assert!(dict.len() > 50000); // Should have many entries
    assert_eq!(dict.language(), "en-us");
    
    // Test lookups
    if let Some(pron) = dict.lookup("hello") {
        println!("Pronunciation for 'hello': {}", pron);
        assert!(pron.phonemes.len() > 3); // Should have several phonemes
    } else {
        panic!("Dictionary should contain 'hello'");
    }
    
    if let Some(pron) = dict.lookup("world") {
        println!("Pronunciation for 'world': {}", pron);
        assert!(pron.phonemes.len() > 3);
    } else {
        panic!("Dictionary should contain 'world'");
    }
}