import torch
from kokoro.model import KModel
from kokoro.pipeline import KPipeline

print('🔬 PYTORCH DURATION LSTM REFERENCE ANALYSIS')
print('=== EXTRACTING EXACT COMPUTATIONAL BEHAVIOR ===')

# Initialize PyTorch system
model = KModel('hexgrad/Kokoro-82M').eval()
pipeline = KPipeline(lang_code='a', model=model)

text = 'Hello world.'
_, tokens = pipeline.g2p(text)
phonemes = pipeline.tokens_to_ps(tokens)
input_ids = list(filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), phonemes)))
input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(model.device)

pack = pipeline.load_voice('af_heart').to(model.device)
ref_s = pack[len(phonemes)-1]

with torch.no_grad():
    input_lengths = torch.full((input_ids.shape[0],), input_ids.shape[-1], device=input_ids.device, dtype=torch.long)
    text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
    text_mask = torch.gt(text_mask+1, input_lengths.unsqueeze(1)).to(model.device)
    
    # Extract PyTorch Duration LSTM behavior
    bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    s = ref_s[:, 128:]
    d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
    x, _ = model.predictor.lstm(d)
    
    print(f'🎯 PYTORCH DURATION LSTM EXACT REFERENCE:')
    print(f'Input shape: {d.shape}')
    print(f'Output shape: {x.shape}')
    print(f'Output std: {x.std().item():.6f}')
    print(f'Output mean: {x.mean().item():.8f}')
    print(f'Output range: [{x.min().item():.6f}, {x.max().item():.6f}]')
    
    # Get duration predictions
    duration = model.predictor.duration_proj(x)
    duration_sigmoid = torch.sigmoid(duration).sum(axis=-1)
    pred_dur = torch.round(duration_sigmoid).clamp(min=1).long().squeeze()
    
    print(f'Duration logits shape: {duration.shape}')
    print(f'Duration predictions: {pred_dur.tolist()}')
    print(f'Total frames: {pred_dur.sum().item()}')
    print()
    print('=== PYTORCH ALIGNMENT VALIDATION TARGET ===')
    print('Our system must match these exact values for proper alignment!')
