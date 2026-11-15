"""Full-length ESM3 inference with bounded query blocks (all keys retained)."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F


def query_chunked_sdpa(original, chunk_size):
    if chunk_size < 1:
        raise ValueError('query chunk size must be positive')
    def attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        if is_causal or dropout_p != 0:
            raise ValueError('Full-length embedding wrapper supports noncausal dropout-free inference only')
        blocks = []
        for start in range(0, query.shape[-2], chunk_size):
            stop = start + chunk_size
            mask = attn_mask
            if mask is not None and mask.ndim >= 2 and mask.shape[-2] != 1:
                mask = mask[..., start:stop, :]
            blocks.append(original(query[..., start:stop, :], key, value,
                                   attn_mask=mask, dropout_p=0.0, is_causal=False, **kwargs))
        return torch.cat(blocks, dim=-2)
    return attention


@contextmanager
def chunked_runtime(chunk_size):
    from src.features import build_esm3_embeddings as build
    from esm.layers.geom_attention import GeometricReasoningOriginalImpl
    geometry_original = GeometricReasoningOriginalImpl.forward
    def geometry(self, s, affine, affine_mask, sequence_id, chain_id):
        if self.mask_and_zero_frameless and not bool(affine_mask.any()):
            return self.out_proj(s.new_zeros((*s.shape[:-1], self.out_proj.in_features)))
        return geometry_original(self, s, affine, affine_mask, sequence_id, chain_id)
    original = F.scaled_dot_product_attention
    fingerprint = build._runtime_fingerprint
    def identity(weights):
        return {**fingerprint(weights), 'query_chunking': {
            'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'query_chunk_size': chunk_size, 'full_key_value_context': True,
            'causal': False, 'dropout': 0.0, 'all_frameless_geometry_exact_zero': True}}
    GeometricReasoningOriginalImpl.forward = geometry
    F.scaled_dot_product_attention = query_chunked_sdpa(original, chunk_size)
    build._runtime_fingerprint = identity
    try:
        yield
    finally:
        GeometricReasoningOriginalImpl.forward = geometry_original
        F.scaled_dot_product_attention = original
        build._runtime_fingerprint = fingerprint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query-chunk-size', type=int, default=512)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--verify-input', help='CSV record_id,sequence for real unmodified versus chunked ESM comparison')
    parser.add_argument('--verification-output')
    args, remaining = parser.parse_known_args(argv)
    if args.threads < 1 or args.query_chunk_size < 1:
        parser.error('threads and query-chunk-size must be positive')
    torch.set_num_threads(args.threads)
    if args.verify_input:
        import pandas as pd
        from src.features.esm3_encoder import ESM3Encoder
        p = argparse.ArgumentParser()
        p.add_argument('--esm-checkpoint', required=True)
        weights = p.parse_args(remaining).esm_checkpoint
        output = Path(args.verification_output)
        if output.exists(): raise FileExistsError(output)
        encoder = ESM3Encoder(device='cpu', weights_path=weights)
        rows = []
        for row in pd.read_csv(args.verify_input).itertuples():
            normal = encoder.encode_sequence(row.record_id, row.sequence)
            with chunked_runtime(args.query_chunk_size):
                chunked = encoder.encode_sequence(row.record_id, row.sequence)
            delta = np.abs(normal.residue_embedding - chunked.residue_embedding)
            passed = np.allclose(normal.residue_embedding, chunked.residue_embedding, rtol=1e-4, atol=1e-4)
            rows.append({'record_id':row.record_id,'length':len(row.sequence),
                         'max_absolute_error':float(delta.max()),'mean_absolute_error':float(delta.mean()),
                         'residue_allclose':bool(passed),'global_allclose':bool(np.allclose(normal.global_embedding,chunked.global_embedding,rtol=1e-4,atol=1e-4))})
            print(json.dumps(rows[-1]), flush=True)
        result={'records':rows,'rtol':1e-4,'atol':1e-4,'query_chunk_size':args.query_chunk_size,
                'input_sha256':hashlib.sha256(Path(args.verify_input).read_bytes()).hexdigest(),
                'checkpoint_sha256':hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
                'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2))
        if not all(r['residue_allclose'] and r['global_allclose'] for r in rows):
            raise ValueError('Real ESM query chunk equivalence failed')
    else:
        from src.features import build_esm3_embeddings as build
        previous = sys.argv
        try:
            sys.argv = [__file__, *remaining]
            with chunked_runtime(args.query_chunk_size):
                build.main()
        finally:
            sys.argv = previous

if __name__ == '__main__':
    main()
