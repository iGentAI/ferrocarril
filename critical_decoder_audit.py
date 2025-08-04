import json

with open('ferrocarril_weights/model/metadata.json') as f:
    metadata = json.load(f)

decoder = metadata['components']['decoder']['parameters']

print('❌ CRITICAL DECODER WEIGHT AUDIT')
print('=' * 50)
print(f'Total decoder parameters: {len(decoder)}')
print()

# Search for the missing weight
target = 'module.generator.conv_pre.weight'
found = target in decoder
print(f'Target weight: {target}')
print(f'Found in metadata: {found}')

if not found:
    print('\n🔍 SEARCHING FOR SIMILAR WEIGHTS:')
    # Find all generator weights
    generator_weights = [w for w in decoder.keys() if 'generator' in w]
    conv_weights = [w for w in decoder.keys() if 'conv' in w]
    
    print(f'\nGenerator weights ({len(generator_weights)}):') 
    for w in sorted(generator_weights)[:10]:
        print(f'  {w}')
    
    print(f'\nConv weights ({len(conv_weights)}):') 
    for w in sorted(conv_weights)[:10]:
        print(f'  {w}')
    
    print('\n❌ MISSING DECODER WEIGHT IS CRITICAL FAILURE')
    print('This will break audio generation completely')
else:
    print('✅ Weight found in metadata')
