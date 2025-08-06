import kokoro.model

print('=== v1.0 KModel Architecture Analysis ===')
kmodel = kokoro.model.KModel()

print('\nTop-level components:')
for name, module in kmodel.named_modules():
    if '.' not in name and name:
        print(f'  {name}: {type(module).__name__}')

print('\nParameter count by component:')
component_counts = {}
for name, param in kmodel.named_parameters():
    component = name.split('.')[0]
    if component not in component_counts:
        component_counts[component] = 0
    component_counts[component] += param.numel()

for comp, count in component_counts.items():
    print(f'  {comp}: {count:,} parameters')

total = sum(component_counts.values())
print(f'\nTotal: {total:,} parameters')
