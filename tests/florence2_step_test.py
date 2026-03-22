"""
2-step manual decoder forward pass test.
Tests if the SA cache from step 1 is correctly passed to step 2.
If step 2 gives different prediction from step 1, cache is working.
If step 2 still predicts <s>, cache is broken.
"""
import sys
sys.path.insert(0, "c:/project/Masters_degree")

import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from src.baseline.model_loader import load_model

cfg = OmegaConf.load("configs/models/florence2_large.yaml")
model, processor = load_model(cfg)
lm = model.language_model

# Determine device from model parameters
device = next(lm.parameters()).device
dtype = next(lm.parameters()).dtype
print(f"Model device: {device}, dtype: {dtype}")

# 768x768 image -> 577 image tokens
img = Image.fromarray(np.full((768, 768, 3), 128, dtype=np.uint8))
question = "What organ is shown?"
prompt = "<VQA>" + question

inputs = processor(text=prompt, images=img, return_tensors="pt")
pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
input_ids = inputs["input_ids"].to(device=device)

# --- Encode ---
with torch.no_grad():
    img_feats = model._encode_image(pixel_values)
    text_embeds = model.get_input_embeddings()(input_ids)
    inputs_embeds, attn_mask = model._merge_input_ids_with_image_features(img_feats, text_embeds)
    enc_out = lm.model.encoder(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        return_dict=True,
    )

enc_hs = enc_out.last_hidden_state
print(f"Encoder: shape={enc_hs.shape}, std={enc_hs.std().item():.4f}")

def top5(logits, processor):
    probs = torch.softmax(logits[0], dim=0)
    vals, ids = torch.topk(probs, 5)
    return [(i.item(), processor.tokenizer.decode([i.item()]), v.item()*100) for i, v in zip(ids, vals)]

# --- Step 1: decoder_start = </s> = 2 ---
with torch.no_grad():
    out1 = lm.model.decoder(
        input_ids=torch.tensor([[2]], device=device),
        encoder_hidden_states=enc_hs,
        encoder_attention_mask=attn_mask,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
    )
pkv1 = out1.past_key_values
print(f"\nStep 1 PKV type: {type(pkv1)}")
if hasattr(pkv1, 'self_attention_cache'):
    sa = pkv1.self_attention_cache
    print(f"  SA cache: {len(sa.layers)} layers, layer[0].keys shape: {sa.layers[0].keys.shape if sa.layers[0].keys is not None else None}")
elif isinstance(pkv1, tuple):
    print(f"  Tuple cache: {len(pkv1)} layers, layer[0][0] shape: {pkv1[0][0].shape}")

logits1 = lm.lm_head(out1.last_hidden_state[:, -1, :])
t5_1 = top5(logits1, processor)
print("\nStep 1 top-5 (input: token 2 = </s>):")
for tid, tok, pct in t5_1:
    print(f"  {tid} {tok!r}: {pct:.2f}%")

step1_tok = t5_1[0][0]  # predicted token

# --- Step 2: feed step 1 prediction ---
with torch.no_grad():
    out2 = lm.model.decoder(
        input_ids=torch.tensor([[step1_tok]], device=device),
        encoder_hidden_states=enc_hs,
        encoder_attention_mask=attn_mask,
        past_key_values=pkv1,
        use_cache=True,
        return_dict=True,
    )
pkv2 = out2.past_key_values

logits2 = lm.lm_head(out2.last_hidden_state[:, -1, :])
t5_2 = top5(logits2, processor)
print(f"\nStep 2 top-5 (input: token {step1_tok} = {processor.tokenizer.decode([step1_tok])!r}):")
for tid, tok, pct in t5_2:
    print(f"  {tid} {tok!r}: {pct:.2f}%")

step2_tok = t5_2[0][0]

# --- Step 3: feed step 2 prediction ---
with torch.no_grad():
    out3 = lm.model.decoder(
        input_ids=torch.tensor([[step2_tok]], device=device),
        encoder_hidden_states=enc_hs,
        encoder_attention_mask=attn_mask,
        past_key_values=pkv2,
        use_cache=True,
        return_dict=True,
    )

logits3 = lm.lm_head(out3.last_hidden_state[:, -1, :])
t5_3 = top5(logits3, processor)
print(f"\nStep 3 top-5 (input: token {step2_tok} = {processor.tokenizer.decode([step2_tok])!r}):")
for tid, tok, pct in t5_3:
    print(f"  {tid} {tok!r}: {pct:.2f}%")

step3_tok = t5_3[0][0]

print(f"\n=== Manual greedy decode: {[2, step1_tok, step2_tok, step3_tok, '...']} ===")
print(f"Decoded: {processor.tokenizer.decode([step1_tok, step2_tok, step3_tok])!r}")
print("\n*** If step2/3 predictions are different from step1, cache is working ***")
print("*** If step2/3 keep predicting same token as step1, SA cache is broken ***")
