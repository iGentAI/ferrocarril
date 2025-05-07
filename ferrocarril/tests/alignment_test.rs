#[cfg(test)]
mod alignment_test {
    use ferrocarril_core::tensor::Tensor;
    use std::error::Error;

    // Test function to create an alignment matrix from durations
    // Similar to the implementation in `ferro_model.rs`
    fn create_alignment_from_durations(durations: &[usize]) -> Tensor<f32> {
        let seq_len = durations.len();
        let total_frames: usize = durations.iter().sum();
        
        // Create indices tensor by repeating position indices according to durations
        let mut indices = Vec::with_capacity(total_frames);
        for (i, &dur) in durations.iter().enumerate() {
            for _ in 0..dur {
                indices.push(i);
            }
        }
        
        // Create alignment matrix [seq_len, total_frames]
        // For every frame position, we set a 1 at the token position it corresponds to
        let mut alignment_data = vec![0.0; seq_len * total_frames];
        for (frame_idx, &token_idx) in indices.iter().enumerate() {
            alignment_data[token_idx * total_frames + frame_idx] = 1.0;
        }
        
        Tensor::from_data(
            alignment_data,
            vec![seq_len, total_frames]
        )
    }
    
    #[test]
    fn test_alignment_creation() -> Result<(), Box<dyn Error>> {
        // Test case 1: Simple durations
        let durations1 = vec![1, 2, 3];
        let alignment1 = create_alignment_from_durations(&durations1);
        
        // Expected shape is [seq_len, sum(durations)] = [3, 6]
        assert_eq!(alignment1.shape(), &[3, 6], "Alignment shape mismatch");
        
        // Verify alignment pattern
        // First token should align with frame 0
        assert_eq!(alignment1[&[0, 0]], 1.0);
        // Second token should align with frames 1 and 2
        assert_eq!(alignment1[&[1, 1]], 1.0);
        assert_eq!(alignment1[&[1, 2]], 1.0);
        // Third token should align with frames 3, 4, and 5
        assert_eq!(alignment1[&[2, 3]], 1.0);
        assert_eq!(alignment1[&[2, 4]], 1.0);
        assert_eq!(alignment1[&[2, 5]], 1.0);
        
        // Test case 2: More complex durations
        let durations2 = vec![2, 1, 4, 3];
        let alignment2 = create_alignment_from_durations(&durations2);
        
        // Expected shape is [seq_len, sum(durations)] = [4, 10]
        assert_eq!(alignment2.shape(), &[4, 10], "Alignment shape mismatch");
        
        // Count number of frames aligned with each token position
        let mut token_frame_counts = vec![0; durations2.len()];
        for t in 0..durations2.len() {
            for f in 0..10 {
                if alignment2[&[t, f]] > 0.5 {
                    token_frame_counts[t] += 1;
                }
            }
        }
        
        // Verify each token has the correct number of aligned frames
        for (i, &dur) in durations2.iter().enumerate() {
            assert_eq!(token_frame_counts[i], dur, 
                      "Token {} should have {} frames, but has {}", 
                      i, dur, token_frame_counts[i]);
        }
        
        // Test case 3: Uneven distribution test
        // This tests the "hard" case where some frames are much longer than others
        let durations3 = vec![1, 10, 1];
        let alignment3 = create_alignment_from_durations(&durations3);
        
        // Expected shape is [seq_len, sum(durations)] = [3, 12]
        assert_eq!(alignment3.shape(), &[3, 12], "Alignment shape mismatch");
        
        // Verify the first token is only aligned with frame 0
        assert_eq!(alignment3[&[0, 0]], 1.0);
        for f in 1..12 {
            assert_eq!(alignment3[&[0, f]], 0.0);
        }
        
        // Verify the middle token (index 1) is aligned with frames 1-10
        for f in 1..11 {
            assert_eq!(alignment3[&[1, f]], 1.0, "Middle token should align with frame {}", f);
        }
        
        // Verify the last token is only aligned with frame 11
        assert_eq!(alignment3[&[2, 11]], 1.0);
        for f in 0..11 {
            assert_eq!(alignment3[&[2, f]], 0.0);
        }
        
        // Test case 4: Verify that every frame is assigned to exactly one token
        let durations4 = vec![3, 2, 5];
        let alignment4 = create_alignment_from_durations(&durations4);
        
        // Expected shape is [seq_len, sum(durations)] = [3, 10]
        assert_eq!(alignment4.shape(), &[3, 10], "Alignment shape mismatch");
        
        // Verify each frame is assigned to exactly one token
        for f in 0..10 {
            let mut sum = 0.0;
            for t in 0..3 {
                sum += alignment4[&[t, f]];
            }
            assert_eq!(sum, 1.0, "Frame {} should be assigned to exactly one token", f);
        }
        
        // Test case 5: Verify it works with single token
        let durations5 = vec![5];
        let alignment5 = create_alignment_from_durations(&durations5);
        
        // Expected shape is [seq_len, sum(durations)] = [1, 5]
        assert_eq!(alignment5.shape(), &[1, 5], "Alignment shape mismatch");
        
        // Verify all frames are assigned to the single token
        for f in 0..5 {
            assert_eq!(alignment5[&[0, f]], 1.0, "All frames should be assigned to the only token");
        }
        
        println!("All alignment tests passed successfully!");
        
        Ok(())
    }
}