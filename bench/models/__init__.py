"""Built-in model presets. Values follow the public HF ``config.json`` files.

Add a model by registering a ``ModelSpec`` here, or pass a path to a
``config.json`` / model directory anywhere a model name is accepted.
"""

from __future__ import annotations

from bench.models.base import (
    AttnVariant,
    ModelRegistry,
    ModelSpec,
    MoESpec,
    from_hf_config,
)

REGISTRY = ModelRegistry()

# --- Qwen ------------------------------------------------------------------
REGISTRY.add(
    ModelSpec(
        name="qwen3-8b",
        family="qwen",
        hf_id="Qwen/Qwen3-8B",
        hidden_size=4096,
        num_layers=36,
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        qk_norm=True,
        intermediate_size=12288,
        vocab_size=151936,
    )
)
REGISTRY.add(
    ModelSpec(
        name="qwen3-30b-a3b",
        family="qwen",
        hf_id="Qwen/Qwen3-30B-A3B",
        hidden_size=2048,
        num_layers=48,
        num_q_heads=32,
        num_kv_heads=4,
        head_dim=128,
        qk_norm=True,
        moe=MoESpec(num_experts=128, topk=8, intermediate_size=768),
        vocab_size=151936,
    )
)
# Qwen3.8-Flash-Next QSA (shapes from demo/shapes.py): sparse decode over the
# indexer budget (2048 selected tokens + tail of the incomplete block).
REGISTRY.add(
    ModelSpec(
        name="qwen3.8-flash-next-qsa",
        family="qwen",
        hidden_size=2560,
        num_layers=48,
        num_q_heads=24,
        num_kv_heads=2,
        head_dim=256,
        qk_norm=True,
        intermediate_size=9216,
        variants=(AttnVariant("qsa_sparse", 48, sparse_topk=2051),),
        notes="attention dims from demo/shapes.py; num_layers and intermediate_size "
        "are placeholders. Indexer/topk/compress stages live in demo/qsa_attention_demo.py",
    )
)

# --- DeepSeek --------------------------------------------------------------
REGISTRY.add(
    ModelSpec(
        name="deepseek-v3",
        family="deepseek",
        hf_id="deepseek-ai/DeepSeek-V3",
        attn_type="mla",
        hidden_size=7168,
        num_layers=61,
        num_q_heads=128,
        kv_lora_rank=512,
        qk_rope_dim=64,
        qk_nope_dim=128,
        q_lora_rank=1536,
        head_dim=192,
        v_head_dim=128,
        intermediate_size=18432,
        moe=MoESpec(num_experts=256, topk=8, intermediate_size=2048, num_shared_experts=1),
        vocab_size=129280,
    )
)
REGISTRY.add(
    ModelSpec(
        name="deepseek-v3.2-dsa",
        family="deepseek",
        hf_id="deepseek-ai/DeepSeek-V3.2-Exp",
        attn_type="mla",
        hidden_size=7168,
        num_layers=61,
        num_q_heads=128,
        kv_lora_rank=512,
        qk_rope_dim=64,
        qk_nope_dim=128,
        q_lora_rank=1536,
        head_dim=192,
        v_head_dim=128,
        intermediate_size=18432,
        moe=MoESpec(num_experts=256, topk=8, intermediate_size=2048, num_shared_experts=1),
        variants=(AttnVariant("dsa_sparse", 61, sparse_topk=2048),),
        vocab_size=129280,
    )
)
# DeepSeek-V4.1-Flash (attention shapes from demo/shapes.py). Single-latent MQA
# with head_dim 512 (448 nope + 64 rope, rope already inside the latent), so
# q/k/v all use the full 512-dim latent: kv_lora_rank=512, qk_rope_dim=0.
# Sparse decode attends to index_topk (512) + window (128) keys.
REGISTRY.add(
    ModelSpec(
        name="deepseek-v4.1-flash",
        family="deepseek",
        attn_type="mla",
        hidden_size=4096,
        num_layers=43,
        num_q_heads=64,
        kv_lora_rank=512,
        qk_rope_dim=0,
        qk_nope_dim=512,
        q_lora_rank=1280,
        head_dim=512,
        intermediate_size=12288,
        moe=MoESpec(num_experts=256, topk=6, intermediate_size=2048, num_shared_experts=1),
        variants=(AttnVariant("dsv4_sparse", 43, sink=True, sparse_topk=640),),
        notes="attention dims from demo/shapes.py; num_layers and MoE sizes are "
        "placeholders, override with the real config.json. Compressor/indexer "
        "stages live in demo/dsv4_attention_demo.py",
    )
)

# --- GLM -------------------------------------------------------------------
REGISTRY.add(
    ModelSpec(
        name="glm-4.5-air",
        family="glm",
        hf_id="zai-org/GLM-4.5-Air",
        hidden_size=4096,
        num_layers=46,
        num_q_heads=96,
        num_kv_heads=8,
        head_dim=128,
        rotary_dim=64,
        moe=MoESpec(num_experts=128, topk=8, intermediate_size=1408, num_shared_experts=1),
        vocab_size=151552,
    )
)
REGISTRY.add(
    ModelSpec(
        name="glm-4-9b",
        family="glm",
        hf_id="THUDM/glm-4-9b-chat",
        hidden_size=4096,
        num_layers=40,
        num_q_heads=32,
        num_kv_heads=2,
        head_dim=128,
        rotary_dim=64,
        intermediate_size=13696,
        vocab_size=151552,
    )
)

# --- GPT-oss ---------------------------------------------------------------
REGISTRY.add(
    ModelSpec(
        name="gpt-oss-20b",
        family="gpt-oss",
        hf_id="openai/gpt-oss-20b",
        hidden_size=2880,
        num_layers=24,
        num_q_heads=64,
        num_kv_heads=8,
        head_dim=64,
        moe=MoESpec(num_experts=32, topk=4, intermediate_size=2880),
        variants=(
            AttnVariant("sliding", 12, window=128, sink=True),
            AttnVariant("full", 12, sink=True),
        ),
        vocab_size=201088,
    )
)
REGISTRY.add(
    ModelSpec(
        name="gpt-oss-120b",
        family="gpt-oss",
        hf_id="openai/gpt-oss-120b",
        hidden_size=2880,
        num_layers=36,
        num_q_heads=64,
        num_kv_heads=8,
        head_dim=64,
        moe=MoESpec(num_experts=128, topk=4, intermediate_size=2880),
        variants=(
            AttnVariant("sliding", 18, window=128, sink=True),
            AttnVariant("full", 18, sink=True),
        ),
        vocab_size=201088,
    )
)

# --- Gemma -----------------------------------------------------------------
REGISTRY.add(
    ModelSpec(
        name="gemma-3-12b",
        family="gemma",
        hf_id="google/gemma-3-12b-it",
        hidden_size=3840,
        num_layers=48,
        num_q_heads=16,
        num_kv_heads=8,
        head_dim=256,
        qk_norm=True,
        intermediate_size=15360,
        variants=(
            AttnVariant("local", 40, window=1024),
            AttnVariant("global", 8),
        ),
        vocab_size=262208,
    )
)
REGISTRY.add(
    ModelSpec(
        name="gemma-2-9b",
        family="gemma",
        hf_id="google/gemma-2-9b-it",
        hidden_size=3584,
        num_layers=42,
        num_q_heads=16,
        num_kv_heads=8,
        head_dim=256,
        intermediate_size=14336,
        variants=(
            AttnVariant("local", 21, window=4096, softcap=50.0),
            AttnVariant("global", 21, softcap=50.0),
        ),
        vocab_size=256000,
    )
)

# Tiny model for CPU smoke tests of the harness itself.
REGISTRY.add(
    ModelSpec(
        name="tiny-gqa",
        family="test",
        hidden_size=256,
        num_layers=2,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=32,
        intermediate_size=512,
        variants=(AttnVariant("local", 1, window=16), AttnVariant("global", 1)),
    )
)
REGISTRY.add(
    ModelSpec(
        name="tiny-mla-moe",
        family="test",
        attn_type="mla",
        hidden_size=256,
        num_layers=2,
        num_q_heads=8,
        kv_lora_rank=64,
        qk_rope_dim=16,
        qk_nope_dim=32,
        head_dim=48,
        moe=MoESpec(num_experts=8, topk=2, intermediate_size=128),
    )
)

__all__ = [
    "REGISTRY",
    "AttnVariant",
    "ModelSpec",
    "MoESpec",
    "from_hf_config",
]
