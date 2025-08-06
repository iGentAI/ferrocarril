#!/usr/bin/env python3
"""
Improved Speech Detection Filter
Uses multiple metrics to properly identify speech vs noise, handling audio that starts with silence.
"""

import sys
import struct
import numpy as np

def analyze_audio_comprehensive(wav_file):
    """Comprehensive speech analysis that handles quiet beginnings."""
    print(f"🔍 COMPREHENSIVE SPEECH ANALYSIS: {wav_file}")
    print("=" * 50)
    
    try:
        # Parse WAV file manually
        with open(wav_file, 'rb') as f:
            # Skip WAV header (44 bytes)
            f.seek(44)
            data = f.read()
        
        # Convert to samples
        samples = struct.unpack('<' + 'h' * (len(data) // 2), data)
        audio_f32 = np.array(samples, dtype=np.float32) / 32767.0
        
        print(f"📊 BASIC PROPERTIES:")
        print(f"   Samples: {len(samples)}")
        print(f"   Duration: {len(samples) / 24000:.3f}s (assuming 24kHz)")
        print(f"   Range: {min(samples)} to {max(samples)}")
        
        # Multiple analysis windows to handle quiet beginnings
        analysis_windows = [
            (0, 1000, "start"),
            (len(samples)//4, len(samples)//4 + 1000, "quarter"),
            (len(samples)//2, len(samples)//2 + 1000, "middle"),
            (max(0, len(samples)-1000), len(samples), "end"),
        ]
        
        max_unique_values = 0
        max_rms = 0.0
        
        print(f"\n🔍 WINDOWED ANALYSIS:")
        
        for start, end, label in analysis_windows:
            if end > len(samples):
                end = len(samples)
            if start >= end:
                continue
                
            window_samples = samples[start:end]
            unique_count = len(set(window_samples))
            rms = np.sqrt(np.mean(np.array(window_samples, dtype=np.float64) ** 2))
            
            print(f"   {label} window [{start}:{end}]: {unique_count} unique, RMS: {rms:.0f}")
            
            max_unique_values = max(max_unique_values, unique_count)
            max_rms = max(max_rms, rms)
        
        # Energy dynamics analysis
        frame_size = len(samples) // 10  # 10 frames
        frame_energies = []
        
        for i in range(0, len(samples) - frame_size, frame_size):
            frame = audio_f32[i:i + frame_size]
            frame_energy = np.mean(frame ** 2)
            frame_energies.append(frame_energy)
        
        frame_energies = np.array(frame_energies)
        energy_variance = np.var(frame_energies) if len(frame_energies) > 1 else 0
        non_zero_frames = np.sum(frame_energies > 1e-6)
        
        print(f"\n📈 TEMPORAL DYNAMICS:")
        print(f"   Frame energy variance: {energy_variance:.8f}")
        print(f"   Non-zero frames: {non_zero_frames}/{len(frame_energies)}")
        print(f"   Energy spread: {non_zero_frames/len(frame_energies)*100:.1f}%")
        
        # Overall content metrics
        overall_rms = np.sqrt(np.mean(audio_f32 ** 2))
        overall_std = np.std(audio_f32)
        zero_crossing_rate = np.sum(np.abs(np.diff(np.sign(audio_f32)))) / (2 * len(audio_f32))
        
        print(f"\n📊 SIGNAL CHARACTERISTICS:")
        print(f"   Overall RMS: {overall_rms:.6f}")
        print(f"   Overall std: {overall_std:.6f}")
        print(f"   Zero crossing rate: {zero_crossing_rate:.6f}")
        print(f"   Max unique in any window: {max_unique_values}")
        print(f"   Max RMS in any window: {max_rms:.0f}")
        
        # Speech likelihood scoring
        score = 0
        reasons = []
        
        # Factor 1: Content variation across windows
        if max_unique_values > 50:
            score += 3
            reasons.append(f"✅ Good variation: {max_unique_values} unique in best window")
        elif max_unique_values > 20:
            score += 2
            reasons.append(f"⚠️ Moderate variation: {max_unique_values} unique in best window")
        else:
            score += 0
            reasons.append(f"❌ Low variation: {max_unique_values} unique in best window")
        
        # Factor 2: Temporal energy dynamics
        if energy_variance > 1e-6:
            score += 2
            reasons.append(f"✅ Temporal variation: energy variance {energy_variance:.8f}")
        else:
            score += 0
            reasons.append(f"❌ No temporal variation: energy variance {energy_variance:.8f}")
        
        # Factor 3: Signal presence
        if overall_std > 0.001:
            score += 2
            reasons.append(f"✅ Signal present: std {overall_std:.6f}")
        elif overall_std > 0.0001:
            score += 1
            reasons.append(f"⚠️ Weak signal: std {overall_std:.6f}")
        else:
            score += 0
            reasons.append(f"❌ Very weak signal: std {overall_std:.6f}")
        
        # Factor 4: Speech-like zero crossing
        if 0.01 < zero_crossing_rate < 0.3:
            score += 2
            reasons.append(f"✅ Speech-like zero crossings: {zero_crossing_rate:.6f}")
        else:
            score += 0
            reasons.append(f"❌ Non-speech zero crossings: {zero_crossing_rate:.6f}")
        
        # Factor 5: Content distribution
        activity_ratio = non_zero_frames / len(frame_energies)
        if activity_ratio > 0.3:
            score += 1
            reasons.append(f"✅ Good activity: {activity_ratio*100:.1f}% frames active")
        else:
            score += 0
            reasons.append(f"❌ Low activity: {activity_ratio*100:.1f}% frames active")
        
        # Classification
        if score >= 7:
            classification = "🎤 HIGHLY LIKELY SPEECH"
        elif score >= 5:
            classification = "🎵 LIKELY SPEECH"
        elif score >= 3:
            classification = "🔊 POSSIBLE SPEECH OR COMPLEX AUDIO"
        elif score >= 1:
            classification = "⚪ WEAK AUDIO SIGNAL"
        else:
            classification = "🔴 LIKELY NOISE OR SILENCE"
        
        print(f"\n🎯 CLASSIFICATION:")
        print(f"   {classification}")
        print(f"   Confidence Score: {score}/10")
        
        print(f"\n📋 REASONING:")
        for reason in reasons:
            print(f"   {reason}")
        
        return {
            'classification': classification,
            'score': score,
            'max_unique_values': max_unique_values,
            'overall_std': overall_std,
            'energy_variance': energy_variance,
            'zero_crossing_rate': zero_crossing_rate,
            'activity_ratio': activity_ratio,
            'reasons': reasons
        }
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 improved_speech_detection.py <wav_file>")
        sys.exit(1)
    
    wav_file = sys.argv[1]
    result = analyze_audio_comprehensive(wav_file)
    
    if result and result['score'] >= 5:
        print(f"\n✅ VERDICT: Audio contains meaningful content")
    else:
        print(f"\n⚠️ VERDICT: Audio may be noise or very quiet")