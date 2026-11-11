"""Unit tests for Speculative Decoding KV-cache slicing and mathematical metrics."""

import torch
from server.kv_cache_manager import KVCacheManager, SessionKVCache
from common.logger import SpeculativeMetrics


def test_kv_cache_cropping_tensor_slicing():
    session = SessionKVCache("test_crop")
    # Simulate 2 layers of (key, value) tensors
    # Shape: [batch=1, num_heads=4, seq_len=10, head_dim=16]
    layers = []
    for _ in range(2):
        k = torch.randn(1, 4, 10, 16)
        v = torch.randn(1, 4, 10, 16)
        layers.append((k, v))
    session.past_key_values = tuple(layers)
    session.current_length = 10
    session.verified_token_ids = list(range(10))

    # Crop to 6
    session.crop_to_length(6)
    assert session.current_length == 6
    assert len(session.verified_token_ids) == 6
    # Verify tensor shapes sliced to seq_len=6
    for layer in session.past_key_values:
        assert layer[0].shape == (1, 4, 6, 16)
        assert layer[1].shape == (1, 4, 6, 16)


def test_kv_cache_manager():
    manager = KVCacheManager(session_ttl_seconds=10)
    sess = manager.create_session("sess_123")
    assert sess.session_id == "sess_123"

    retrieved = manager.get_session("sess_123")
    assert retrieved is not None
    assert retrieved.session_id == "sess_123"

    freed = manager.remove_session("sess_123")
    assert freed is True
    assert manager.get_session("sess_123") is None


def test_speculative_metrics():
    metrics = SpeculativeMetrics()
    # Step 1: 4 drafted, 3 accepted
    log1 = metrics.record_step(step=1, drafted=4, accepted=3, corrective_token_text="next")
    assert "3/4" in log1
    assert "75.0%" in log1

    # Step 2: 4 drafted, 4 accepted
    log2 = metrics.record_step(step=2, drafted=4, accepted=4, corrective_token_text="word")
    assert "4/4" in log2
    assert "100.0%" in log2

    assert metrics.total_drafted == 8
    assert metrics.total_accepted == 7
    assert round(metrics.cumulative_acceptance_rate, 3) == round(7 / 8, 3)
    # Effective tokens = (7 + 2) / 2 = 4.5
    assert metrics.estimated_speedup == 4.5

    summary = metrics.summary()
    assert "Speculative Decoding Summary" in summary
    assert "87.5%" in summary


if __name__ == "__main__":
    test_kv_cache_cropping_tensor_slicing()
    test_kv_cache_manager()
    test_speculative_metrics()
    print("All speculative math and KV-cache tests passed!")

