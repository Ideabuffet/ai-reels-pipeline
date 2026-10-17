#!/usr/bin/env python3
"""WanVideoWrapper's multitalk/wav2vec2.py breaks on transformers 5.x: the encoder no longer returns
hidden_states, so InfiniteTalk fails with `hidden_states = None`. Run the encoder layers manually,
exactly like the 4.x Wav2Vec2Encoder (pos_conv -> layer_norm -> dropout -> layers). Idempotent.

usage: patch_wav2vec.py <ComfyUI-WanVideoWrapper/multitalk/wav2vec2.py>
"""
import sys

MARK = "# transformers>=5 no longer returns hidden_states"
OLD = """        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]
"""
NEW = """        # transformers>=5 no longer returns hidden_states from the encoder here; run layers manually
        # (same as the 4.x Wav2Vec2Encoder: pos_conv -> layer_norm -> dropout -> layers)
        from types import SimpleNamespace
        enc = self.encoder
        hs = hidden_states + enc.pos_conv_embed(hidden_states)
        hs = enc.dropout(enc.layer_norm(hs))
        all_hs = (hs,)
        for layer in enc.layers:
            out = layer(hs, attention_mask=None)
            hs = out[0] if isinstance(out, (tuple, list)) else out
            all_hs += (hs,)
        encoder_outputs = SimpleNamespace(hidden_states=all_hs, attentions=None)
        hidden_states = hs
        return_dict = True
"""

path = sys.argv[1]
src = open(path).read()
if MARK in src:
    print("wav2vec2 patch: already applied")
    sys.exit(0)
n = src.count(OLD)
if not n:
    sys.exit("wav2vec2 patch: pattern not found — WanVideoWrapper changed, re-check the patch")
open(path, "w").write(src.replace(OLD, NEW))
print(f"wav2vec2 patch: applied ({n} place)")
