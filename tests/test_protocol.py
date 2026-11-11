"""Unit tests for SpecRoute Protocol Models."""

import pytest
from common.protocol import (
    TokenizerCheckRequest,
    TokenizerCheckResponse,
    SessionInitRequest,
    SessionInitResponse,
    VerifyRequest,
    VerifyResponse,
    SessionCloseRequest,
    SessionCloseResponse,
)


def test_tokenizer_check_payload():
    req = TokenizerCheckRequest(
        tokenizer_id="Qwen/Qwen2.5-0.5B-Instruct",
        vocab_size=151646,
        test_phrase="Hello test",
        test_token_ids=[100, 200, 300],
    )
    json_data = req.model_dump()
    assert json_data["vocab_size"] == 151646
    assert json_data["test_token_ids"] == [100, 200, 300]

    resp = TokenizerCheckResponse(
        status="ok",
        synchronized=True,
        server_tokenizer_id="Qwen/Qwen2.5-7B-Instruct",
        server_vocab_size=151646,
        matches_vocab_size=True,
        matches_test_ids=True,
        message="Synchronized",
    )
    assert resp.synchronized is True


def test_session_init_payload():
    req = SessionInitRequest(
        session_id="test_sess_01",
        prompt_token_ids=[1, 2, 3, 4, 5],
        temperature=0.0,
        top_p=0.95,
    )
    assert req.session_id == "test_sess_01"
    assert len(req.prompt_token_ids) == 5

    resp = SessionInitResponse(
        session_id="test_sess_01",
        status="ok",
        prefill_length=5,
        first_token_id=42,
        is_eos=False,
    )
    assert resp.first_token_id == 42
    assert resp.is_eos is False


def test_verify_payload():
    req = VerifyRequest(
        session_id="test_sess_01",
        step_index=1,
        draft_tokens=[10, 20, 30, 40],
        temperature=0.0,
    )
    assert len(req.draft_tokens) == 4

    resp = VerifyResponse(
        session_id="test_sess_01",
        step_index=1,
        accepted_tokens=[10, 20, 30],
        accepted_count=3,
        rejection_index=3,
        corrective_token=99,
        acceptance_rate=0.75,
        target_kv_len=8,
        is_eos=False,
    )
    assert resp.accepted_count == 3
    assert resp.corrective_token == 99
    assert resp.acceptance_rate == 0.75


if __name__ == "__main__":
    test_tokenizer_check_payload()
    test_session_init_payload()
    test_verify_payload()
    print("All protocol tests passed!")

