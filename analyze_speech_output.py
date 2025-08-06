#!/usr/bin/env python3
"""
Statistical Analysis of Neural Speech Output
Verifies that generated audio contains real speech characteristics.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import struct
import sys

def read_wav_manual(filepath):
    """Read WAV file manually to handle any format issues."""
    with open(filepath, 'rb') as f:
        # Read WAV header
        riff = f.read(4)
        if riff != b'RIFF':
            raise ValueError(f"Not a valid WAV file: {riff}")
        
        file_size = struct.unpack('<I', f.read(4))[0]
        wave = f.read(4)
        if wave != b'WAVE':
            raise ValueError(f"Not a WAVE file: {wave}")
        
        # Find fmt chunk
        fmt_chunk = f.read(4)
        if fmt_chunk != b'fmt ':
            raise ValueError(f"Expected fmt chunk, got: {fmt_chunk}")
        
        fmt_size = struct.unpack('<I', f.read(4))[0]
        audio_format = struct.unpack('<H', f.read(2))[0]
        num_channels = struct.unpack('<H', f.read(2))[0]
        sample_rate = struct.unpack('<I', f.read(4))[0]
        byte_rate = struct.unpack('<I', f.read(4))[0]
        block_align = struct.unpack('<H', f.read(2))[0]
        bits_per_sample = struct.unpack('<H', f.read(2))[0]
        
        # Skip any extra fmt data
        f.read(fmt_size - 16)
        
        # Find data chunk
        data_chunk = f.read(4)
        if data_chunk != b'data':
            raise ValueError(f"Expected data chunk, got: {data_chunk}")
        
        data_size = struct.unpack('<I', f.read(4))[0]
        
        # Read audio data
        audio_bytes = f.read(data_size)
        
        # Convert to numpy array
        if bits_per_sample == 16:
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0
        else:
            raise ValueError(f"Unsupported bit depth: {bits_per_sample}")
        
        return audio_data, sample_rate, {
            'num_channels': num_channels,
            'bits_per_sample': bits_per_sample,
            'duration': len(audio_data) / sample_rate
        }

def analyze_speech_characteristics(audio_data, sample_rate):
    """Analyze audio for speech-like characteristics."""
    results = {}
    
    # Basic statistics
    results['sample_count'] = len(audio_data)
    results['duration_seconds'] = len(audio_data) / sample_rate
    results['sample_rate'] = sample_rate
    results['amplitude_range'] = (float(np.min(audio_data)), float(np.max(audio_data)))
    results['rms_amplitude'] = float(np.sqrt(np.mean(audio_data**2)))
    results['zero_crossing_rate'] = float(np.sum(np.abs(np.diff(np.sign(audio_data)))) / (2 * len(audio_data)))
    
    # Check for silence or constant values
    results['is_mostly_silent'] = results['rms_amplitude'] < 0.001
    results['amplitude_variance'] = float(np.var(audio_data))
    results['dynamic_range_db'] = 20 * np.log10(np.max(np.abs(audio_data)) / (np.std(audio_data) + 1e-8))
    
    # Spectral analysis
    freqs, times, spectrogram = signal.spectrogram(
        audio_data, sample_rate, nperseg=512, noverlap=256
    )
    
    # Speech frequency analysis (80Hz - 8kHz is typical for speech)
    speech_freq_mask = (freqs >= 80) & (freqs <= 8000)
    speech_energy = np.sum(spectrogram[speech_freq_mask, :])
    total_energy = np.sum(spectrogram)
    results['speech_frequency_ratio'] = float(speech_energy / total_energy) if total_energy > 0 else 0
    
    # Formant-like structure detection (speech typically has formants around 500Hz, 1500Hz, 2500Hz)
    formant_windows = [
        (300, 700),   # F1 region
        (1000, 2000), # F2 region  
        (2000, 3500), # F3 region
    ]
    
    formant_energies = []
    for f_low, f_high in formant_windows:
        formant_mask = (freqs >= f_low) & (freqs <= f_high)
        formant_energy = np.sum(spectrogram[formant_mask, :])
        formant_energies.append(formant_energy)
    
    results['formant_energy_distribution'] = [float(e) for e in formant_energies]
    results['has_formant_structure'] = any(e > total_energy * 0.1 for e in formant_energies)
    
    # Temporal variation analysis (speech has natural rhythm and pauses)
    frame_size = sample_rate // 50  # 20ms frames
    frame_energies = []
    for i in range(0, len(audio_data) - frame_size, frame_size):
        frame = audio_data[i:i + frame_size]
        frame_energy = np.mean(frame**2)
        frame_energies.append(frame_energy)
    
    frame_energies = np.array(frame_energies)
    results['frame_energy_variance'] = float(np.var(frame_energies))
    results['frame_energy_std'] = float(np.std(frame_energies))
    results['has_temporal_variation'] = results['frame_energy_variance'] > 0.0001
    
    # Peak detection for syllable-like structure
    peaks, _ = signal.find_peaks(frame_energies, height=np.max(frame_energies) * 0.1)
    results['estimated_syllable_count'] = len(peaks)
    results['syllable_rate_per_second'] = len(peaks) / results['duration_seconds']
    
    # Check for pure tone characteristics (synthetic audio often has pure tones)
    fft = np.fft.fft(audio_data)
    fft_magnitude = np.abs(fft[:len(fft)//2])
    fft_freqs = np.fft.fftfreq(len(audio_data), 1/sample_rate)[:len(fft)//2]
    
    # Find dominant frequency
    dominant_freq_idx = np.argmax(fft_magnitude)
    dominant_freq = fft_freqs[dominant_freq_idx]
    results['dominant_frequency_hz'] = float(dominant_freq)
    
    # Check if dominated by a single frequency (sign of synthetic tone)
    freq_spread = np.sum(fft_magnitude > np.max(fft_magnitude) * 0.1)
    results['frequency_spread'] = int(freq_spread)
    results['is_likely_pure_tone'] = freq_spread < 10 and results['dominant_frequency_hz'] > 200
    
    return results

def classify_audio_type(results):
    """Classify whether audio appears to be speech, music, noise, or synthetic."""
    score = 0
    reasons = []
    
    # Positive indicators for speech
    if not results['is_mostly_silent']:
        score += 2
        reasons.append("✅ Not silent")
    else:
        reasons.append("❌ Mostly silent")
    
    if results['speech_frequency_ratio'] > 0.6:
        score += 2
        reasons.append(f"✅ Good speech frequency content ({results['speech_frequency_ratio']:.2f})")
    else:
        reasons.append(f"⚠️ Low speech frequency content ({results['speech_frequency_ratio']:.2f})")
    
    if results['has_formant_structure']:
        score += 2
        reasons.append("✅ Has formant-like structure")
    else:
        reasons.append("❌ No formant structure")
    
    if results['has_temporal_variation']:
        score += 1
        reasons.append("✅ Has temporal variation")
    else:
        reasons.append("❌ No temporal variation")
    
    if results['syllable_rate_per_second'] > 2 and results['syllable_rate_per_second'] < 10:
        score += 2
        reasons.append(f"✅ Realistic syllable rate ({results['syllable_rate_per_second']:.1f}/sec)")
    else:
        reasons.append(f"⚠️ Unusual syllable rate ({results['syllable_rate_per_second']:.1f}/sec)")
    
    if not results['is_likely_pure_tone']:
        score += 1
        reasons.append("✅ Not a pure tone")
    else:
        reasons.append("❌ Appears to be pure tone")
    
    if results['dynamic_range_db'] > 10:
        score += 1
        reasons.append(f"✅ Good dynamic range ({results['dynamic_range_db']:.1f}dB)")
    else:
        reasons.append(f"⚠️ Low dynamic range ({results['dynamic_range_db']:.1f}dB)")
    
    # Classification
    if score >= 8:
        classification = "🎤 HIGHLY LIKELY SPEECH"
    elif score >= 6:
        classification = "🎵 LIKELY SPEECH OR COMPLEX AUDIO"
    elif score >= 4:
        classification = "🔊 COMPLEX AUDIO (uncertain)"
    elif score >= 2:
        classification = "⚪ SIMPLE AUDIO PATTERN"
    else:
        classification = "🔴 LIKELY SYNTHETIC OR NOISE"
    
    return classification, score, reasons

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_speech_output.py <wav_file>")
        sys.exit(1)
    
    wav_file = sys.argv[1]
    
    try:
        print(f"🎧 Analyzing Neural Speech Output: {wav_file}")
        print("=" * 60)
        
        # Load and analyze audio
        audio_data, sample_rate, metadata = read_wav_manual(wav_file)
        results = analyze_speech_characteristics(audio_data, sample_rate)
        classification, score, reasons = classify_audio_type(results)
        
        # Display results
        print(f"\n📊 BASIC PROPERTIES:")
        print(f"   Duration: {results['duration_seconds']:.2f}s")
        print(f"   Sample Rate: {results['sample_rate']}Hz")
        print(f"   Sample Count: {results['sample_count']}")
        print(f"   Amplitude Range: {results['amplitude_range'][0]:.3f} to {results['amplitude_range'][1]:.3f}")
        print(f"   RMS Amplitude: {results['rms_amplitude']:.4f}")
        print(f"   Dynamic Range: {results['dynamic_range_db']:.1f}dB")
        
        print(f"\n🔍 SPEECH CHARACTERISTICS:")
        print(f"   Speech Frequency Content: {results['speech_frequency_ratio']:.2%}")
        print(f"   Zero Crossing Rate: {results['zero_crossing_rate']:.4f}")
        print(f"   Formant Structure: {'Yes' if results['has_formant_structure'] else 'No'}")
        print(f"   Temporal Variation: {'Yes' if results['has_temporal_variation'] else 'No'}")
        print(f"   Estimated Syllables: {results['estimated_syllable_count']}")
        print(f"   Syllable Rate: {results['syllable_rate_per_second']:.1f}/second")
        
        print(f"\n🎛️ SYNTHETIC DETECTION:")
        print(f"   Dominant Frequency: {results['dominant_frequency_hz']:.1f}Hz")
        print(f"   Frequency Spread: {results['frequency_spread']} bins")
        print(f"   Pure Tone Likelihood: {'High ❌' if results['is_likely_pure_tone'] else 'Low ✅'}")
        
        print(f"\n🎯 CLASSIFICATION:")
        print(f"   {classification}")
        print(f"   Confidence Score: {score}/10")
        
        print(f"\n📋 DETAILED REASONING:")
        for reason in reasons:
            print(f"   {reason}")
        
        # Generate summary verdict
        print(f"\n" + "=" * 60)
        if score >= 7:
            print("🎉 VERDICT: HIGH CONFIDENCE SPEECH SYNTHESIS")
            print("   The generated audio exhibits strong speech-like characteristics.")
            print("   This appears to be authentic neural TTS output, not synthetic generation.")
        elif score >= 5:
            print("✅ VERDICT: LIKELY SPEECH SYNTHESIS")  
            print("   The generated audio shows speech-like properties.")
            print("   System appears to be functioning as a real TTS engine.")
        else:
            print("⚠️  VERDICT: INCONCLUSIVE OR SYNTHETIC")
            print("   The generated audio may be synthetic or corrupted.")
            print("   Further investigation needed.")
        
        # Export analysis for web display
        with open('analysis_results.txt', 'w') as f:
            f.write(f"Neural Speech Analysis Results\n")
            f.write(f"============================\n\n")
            f.write(f"Classification: {classification}\n")
            f.write(f"Confidence Score: {score}/10\n\n")
            f.write(f"Duration: {results['duration_seconds']:.2f}s\n")
            f.write(f"Speech Frequency Content: {results['speech_frequency_ratio']:.2%}\n")
            f.write(f"Syllable Rate: {results['syllable_rate_per_second']:.1f}/sec\n")
            f.write(f"Dynamic Range: {results['dynamic_range_db']:.1f}dB\n\n")
            f.write("Detailed Analysis:\n")
            for reason in reasons:
                f.write(f"  {reason}\n")
        
        print(f"\n📄 Analysis exported to: analysis_results.txt")
        print(f"🌐 Both files available at: https://lgpssyrs72bqm6tr.preview.dev.igent.ai/")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()