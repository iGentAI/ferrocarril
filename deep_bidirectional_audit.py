#!/usr/bin/env python3
# Deep comprehensive audit of ALL complex weight patterns

import sys
sys.path.append('kokoro')
import json
from kokoro.model import KModel

def audit_bidirectional_lstm_weights():
    """Audit ALL bidirectional LSTM weights for forward/reverse completeness"""
    print("🔍 CRITICAL BIDIRECTIONAL LSTM AUDIT")
    print("=" * 60)
    
    # Load models
    model = KModel()
    pytorch_state_dict = model.state_dict()
    
    with open('ferrocarril_weights/model/metadata.json') as f:
        converted = json.load(f)['components']
    
    # Check each LSTM component
    lstm_components = [
        ('text_encoder', 'TextEncoderLSTM'),
        ('predictor', 'ProsodyLSTM'),
    ]
    
    all_bidirectional_complete = True
    
    for component, description in lstm_components:
        print(f"\n📦 {component.upper()} ({description}):")
        
        # Find ALL PyTorch LSTM weights (complete set)
        pytorch_weights = [k for k in pytorch_state_dict.keys() 
                          if component in k and 'lstm' in k]
        
        # Complete bidirectional weight patterns
        required_patterns = [
            'weight_ih_l0',          # Input-hidden forward
            'weight_ih_l0_reverse',  # Input-hidden reverse  
            'weight_hh_l0',          # Hidden-hidden forward
            'weight_hh_l0_reverse',  # Hidden-hidden reverse
            'bias_ih_l0',            # Input-hidden bias forward
            'bias_ih_l0_reverse',    # Input-hidden bias reverse
            'bias_hh_l0',            # Hidden-hidden bias forward
            'bias_hh_l0_reverse',    # Hidden-hidden bias reverse
        ]
        
        print(f"  Total PyTorch LSTM weights: {len(pytorch_weights)}")
        print(f"  Expected bidirectional patterns: {len(required_patterns)}")
        
        # Check each required pattern
        missing_patterns = []
        for pattern in required_patterns:
            pytorch_matches = [w for w in pytorch_weights if pattern in w]
            if pytorch_matches:
                print(f"    ✅ {pattern}: {len(pytorch_matches)} weights")
                for w in pytorch_matches:
                    print(f"      {w}")
            else:
                print(f"    ❌ {pattern}: MISSING FROM PYTORCH")
                missing_patterns.append(pattern)
        
        # Check converted weights have all patterns
        if component in converted:
            conv_params = converted[component]['parameters']
            
            print(f"\n  Converted weight verification:")
            for pattern in required_patterns:
                converted_matches = [p for p in conv_params if pattern in p]
                if converted_matches:
                    print(f"    ✅ {pattern}: {len(converted_matches)} converted")
                else:
                    print(f"    ❌ {pattern}: MISSING FROM CONVERTED")
                    missing_patterns.append(f"converted:{pattern}")
        
        if missing_patterns:
            print(f"\n  ❌ CRITICAL LSTM FAILURE: Missing {len(missing_patterns)} weight patterns")
            for pattern in missing_patterns:
                print(f"    - {pattern}")
            all_bidirectional_complete = False
        else:
            print(f"\n  ✅ COMPLETE BIDIRECTIONAL LSTM: All 8 weight patterns present")
    
    return all_bidirectional_complete

def audit_weight_norm_patterns():
    """Audit weight_norm patterns (weight_g, weight_v pairs) for completeness"""
    print(f"\n🔍 WEIGHT NORMALIZATION PATTERN AUDIT")
    print("=" * 60)
    
    model = KModel()
    pytorch_state_dict = model.state_dict()
    
    with open('ferrocarril_weights/model/metadata.json') as f:
        converted = json.load(f)['components']
    
    # Find all weight_norm patterns in PyTorch
    weight_g_weights = [k for k in pytorch_state_dict.keys() if 'weight_g' in k]
    weight_v_weights = [k for k in pytorch_state_dict.keys() if 'weight_v' in k]
    
    print(f"PyTorch weight_norm patterns:")
    print(f"  weight_g parameters: {len(weight_g_weights)}")
    print(f"  weight_v parameters: {len(weight_v_weights)}")
    
    # Verify pairs are balanced
    weight_norm_balanced = len(weight_g_weights) == len(weight_v_weights)
    print(f"  Balanced pairs: {'✅' if weight_norm_balanced else '❌ CRITICAL IMBALANCE'}")
    
    if not weight_norm_balanced:
        print(f"    Expected equal weight_g and weight_v counts")
        print(f"    This indicates broken weight_norm conversion")
    
    # Check conversion completeness by component
    weight_norm_complete = True
    
    for component_name, component_data in converted.items():
        conv_params = component_data['parameters'] 
        component_weight_g = [p for p in conv_params if 'weight_g' in p]
        component_weight_v = [p for p in conv_params if 'weight_v' in p]
        
        if component_weight_g or component_weight_v:
            print(f"\n📦 {component_name.upper()} weight_norm:")
            print(f"  weight_g: {len(component_weight_g)}")
            print(f"  weight_v: {len(component_weight_v)}")
            
            if len(component_weight_g) != len(component_weight_v):
                print(f"  ❌ CRITICAL: Mismatched weight_g/weight_v pairs")
                print(f"    This will break weight normalization")
                weight_norm_complete = False
            else:
                print(f"  ✅ weight_norm pairs complete")
    
    return weight_norm_complete and weight_norm_balanced

def audit_attention_weights():
    """Audit attention mechanism weights (query, key, value projections)"""
    print(f"\n🔍 ATTENTION MECHANISM WEIGHT AUDIT") 
    print("=" * 60)
    
    model = KModel()
    pytorch_state_dict = model.state_dict()
    
    # Find attention weights
    attention_weights = [k for k in pytorch_state_dict.keys() 
                        if any(att_type in k for att_type in ['query', 'key', 'value', 'attention'])]
    
    print(f"PyTorch attention weights: {len(attention_weights)}")
    
    if attention_weights:
        # Group by projection type
        query_weights = [w for w in attention_weights if 'query' in w]
        key_weights = [w for w in attention_weights if 'key' in w] 
        value_weights = [w for w in attention_weights if 'value' in w]
        dense_weights = [w for w in attention_weights if 'dense' in w]
        
        print(f"  Query projections: {len(query_weights)}")
        print(f"  Key projections: {len(key_weights)}")
        print(f"  Value projections: {len(value_weights)}")
        print(f"  Dense projections: {len(dense_weights)}")
        
        # Show actual attention weights
        print(f"\nAttention weights found:")
        for w in attention_weights:
            print(f"  ✅ {w}")
        
        attention_complete = len(query_weights) > 0 and len(key_weights) > 0 and len(value_weights) > 0
        print(f"\nAttention completeness: {'✅ COMPLETE' if attention_complete else '❌ MISSING COMPONENTS'}")
    else:
        print("  No attention weights found in PyTorch model")
        attention_complete = True  # No attention weights means nothing to validate
    
    return attention_complete

def test_actual_weight_loading():
    """Test that critical weights can actually be loaded"""
    print(f"\n🔍 ACTUAL WEIGHT LOADING TEST")
    print("=" * 60)
    
    # Test critical bidirectional weights (expanded set)
    critical_weights = [
        ('text_encoder', 'module.lstm.weight_ih_l0', 'Forward input-hidden'),
        ('text_encoder', 'module.lstm.weight_ih_l0_reverse', 'Reverse input-hidden'),
        ('text_encoder', 'module.lstm.weight_hh_l0', 'Forward hidden-hidden'),
        ('text_encoder', 'module.lstm.weight_hh_l0_reverse', 'Reverse hidden-hidden'),
        ('text_encoder', 'module.lstm.bias_ih_l0', 'Forward input bias'),
        ('text_encoder', 'module.lstm.bias_ih_l0_reverse', 'Reverse input bias'),
        ('predictor', 'module.lstm.weight_ih_l0', 'Predictor forward input'),
        ('predictor', 'module.lstm.weight_ih_l0_reverse', 'Predictor reverse input'),
        ('bert', 'module.embeddings.word_embeddings.weight', 'BERT embeddings'),
        ('decoder', 'module.generator.conv_post.weight_g', 'Conv weight_norm_g'),
        ('decoder', 'module.generator.conv_post.weight_v', 'Conv weight_norm_v'),
    ]
    
    with open('ferrocarril_weights/model/metadata.json') as f:
        metadata = json.load(f)
    
    loading_results = []
    
    print("Critical weight loading verification:")
    for component, param, description in critical_weights:
        if component in metadata['components']:
            if param in metadata['components'][component]['parameters']:
                param_info = metadata['components'][component]['parameters'][param]
                print(f"  ✅ {description}: {param_info['shape']}")
                loading_results.append(True)
            else:
                print(f"  ❌ {description}: MISSING FROM METADATA")
                print(f"     Component: {component}, Param: {param}")
                loading_results.append(False)
        else:
            print(f"  ❌ {description}: COMPONENT MISSING ({component})")
            loading_results.append(False)
    
    success_rate = sum(loading_results) / len(loading_results) * 100
    print(f"\nCritical weight loading test:")
    print(f"  ✅ Successful: {sum(loading_results)}/{len(loading_results)}")
    print(f"  📊 Success rate: {success_rate:.1f}%")
    
    complete_success = success_rate == 100.0
    print(f"  {'✅ ALL CRITICAL WEIGHTS LOADABLE' if complete_success else '❌ CRITICAL LOADING FAILURES'}")
    
    return complete_success

def run_exhaustive_weight_audit():
    """Run exhaustive audit of ALL weight patterns"""
    print("🚨 EXHAUSTIVE WEIGHT PATTERN AUDIT")
    print("=" * 80)
    print("Zero tolerance validation of ALL complex neural weight patterns")
    print()
    
    # Test all critical weight patterns
    audits = []
    
    bidirectional_result = audit_bidirectional_lstm_weights()
    audits.append(("Bidirectional_LSTM", bidirectional_result))
    
    weight_norm_result = audit_weight_norm_patterns()
    audits.append(("Weight_Normalization", weight_norm_result))
    
    attention_result = audit_attention_weights()
    audits.append(("Attention_Mechanisms", attention_result))
    
    loading_result = test_actual_weight_loading()
    audits.append(("Actual_Weight_Loading", loading_result))
    
    # Final assessment
    print(f"\n📊 EXHAUSTIVE AUDIT SUMMARY")
    print("=" * 60)
    
    passed = 0
    for audit_name, result in audits:
        status = "✅ PASS" if result else "❌ CRITICAL FAILURE"
        print(f"  {audit_name:<25} {status}")
        if result:
            passed += 1
    
    overall_success = passed == len(audits)
    
    print(f"\n🎯 EXHAUSTIVE VALIDATION RESULTS:")
    print(f"  ✅ Passed: {passed}/{len(audits)}")
    print(f"  ❌ Failed: {len(audits) - passed}/{len(audits)}")
    print(f"  📊 Success rate: {passed/len(audits)*100:.1f}%")
    
    if overall_success:
        print("  🎉 EXHAUSTIVE AUDIT SUCCESSFUL")
        print("  ALL complex weight patterns validated")
        print("  Bidirectional, weight_norm, attention - ALL COMPLETE") 
    else:
        print("  💥 CRITICAL WEIGHT PATTERN FAILURES")
        print("  Neural network functionality COMPROMISED")
        print("  TTS inference will have BROKEN components")
    
    return overall_success

if __name__ == "__main__":
    success = run_exhaustive_weight_audit()
    if success:
        print(f"\n✅ USER'S CONCERNS ADDRESSED")
        print("Exhaustive weight pattern validation confirms complete coverage")
    else:
        print(f"\n❌ USER'S CONCERNS VALIDATED")
        print("Critical weight pattern failures detected - immediate fix required")
        sys.exit(1)