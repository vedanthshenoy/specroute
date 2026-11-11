"""Unit tests for Tokenizer Synchronization and Parity Validation."""

from common.tokenizer_sync import SynchronizedTokenizer
from common.protocol import TokenizerCheckResponse


def test_tokenizer_encoding_decoding():
    tok = SynchronizedTokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    text = "Hello, speculative decoding!"
    token_ids = tok.encode(text)
    assert len(token_ids) > 0
    decoded = tok.decode(token_ids)
    assert "speculative" in decoded.lower() or "hello" in decoded.lower()


def test_chat_template():
    tok = SynchronizedTokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    messages = [
        {"role": "system", "content": "You are a fast AI."},
        {"role": "user", "content": "Explain hybrid inference."},
    ]
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True)
    assert "user" in prompt
    assert "Explain hybrid inference." in prompt
    assert "assistant" in prompt


def test_tokenizer_parity_verification():
    tok = SynchronizedTokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    req = tok.generate_check_request()
    assert req.vocab_size == tok.vocab_size
    assert len(req.test_token_ids) > 0

    # Self-validation
    resp = tok.validate_check_request(req)
    assert resp.synchronized is True
    assert resp.matches_vocab_size is True
    assert resp.matches_test_ids is True

    # Validate against matching response
    assert tok.verify_remote_parity(resp) is True

    # Validate mismatch detection
    mismatch_resp = TokenizerCheckResponse(
        status="mismatch",
        synchronized=False,
        server_tokenizer_id="OtherModel",
        server_vocab_size=32000,
        matches_vocab_size=False,
        matches_test_ids=False,
        message="Vocab size mismatch",
    )
    assert tok.verify_remote_parity(mismatch_resp) is False


if __name__ == "__main__":
    test_tokenizer_encoding_decoding()
    test_chat_template()
    test_tokenizer_parity_verification()
    print("All tokenizer sync tests passed!")

