import json

with open('ferrocarril_weights/model/metadata.json') as f:
    metadata = json.load(f)

print('🔍 MISSING COMPONENT WEIGHT ANALYSIS')
print()

# Check for Conv1d weights
conv_count = 0
linear_count = 0
adain_count = 0
for comp_name, comp_data in metadata['components'].items():
    for param_name, param_info in comp_data['parameters'].items():
        shape = param_info['shape']
        if len(shape) == 3:  # Conv1d weights
            conv_count += 1
        elif len(shape) == 2 and 'fc.weight' in param_name:  # AdaIN fc weights
            adain_count += 1
        elif len(shape) == 2 and ('weight' in param_name and 'lstm' not in param_name):
            linear_count += 1

print(f'Conv1d weights (3D tensors): {conv_count}')
print(f'Linear weights (2D tensors): {linear_count}')
print(f'AdaIN fc weights: {adain_count}')
print()

# Find examples
print('EXAMPLES:')
for comp_name, comp_data in metadata['components'].items():
    conv_examples = [p for p, info in comp_data['parameters'].items() if len(info['shape']) == 3][:2]
    adain_examples = [p for p, info in comp_data['parameters'].items() if 'fc.weight' in p][:2]
    
    if conv_examples:
        print(f'  {comp_name} Conv1d examples: {conv_examples}')
    if adain_examples:
        print(f'  {comp_name} AdaIN examples: {adain_examples}')
