"""Configuration settings loader for SpecRoute Hybrid Speculative Decoding."""

import os
from pathlib import Path
from typing import Optional, Literal
import yaml
from pydantic import BaseModel, Field


class ModelsConfig(BaseModel):
    draft_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    target_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    local_engine_backend: Literal["simulation", "huggingface", "llama_cpp", "ollama"] = "simulation"
    local_device: str = "cpu"


class SpeculativeConfig(BaseModel):
    lookahead_k: int = Field(default=4, ge=1, le=16)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_new_tokens: int = Field(default=512, ge=1, le=8192)
    smooth_streaming: bool = True
    token_pacing_ms: int = Field(default=12, ge=0, le=100)


class RemoteServerConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8001"
    use_websocket: bool = True
    ws_path: str = "/v1/speculative/ws"
    http_verify_path: str = "/v1/speculative/verify"
    timeout_seconds: float = 15.0
    auth_token: Optional[str] = "specroute-cloud-secret"

    @property
    def http_verify_url(self) -> str:
        base = self.base_url.rstrip("/")
        path = self.http_verify_path.lstrip("/")
        return f"{base}/{path}"

    @property
    def ws_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.startswith("http://"):
            base = "ws://" + base[7:]
        elif base.startswith("https://"):
            base = "wss://" + base[8:]
        elif not base.startswith("ws://") and not base.startswith("wss://"):
            base = "ws://" + base
        path = self.ws_path.lstrip("/")
        return f"{base}/{path}"


class LocalProxyConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    model_name: str = "specroute-hybrid-qwen"
    api_key: str = "specroute-local-token"


class MonitoringConfig(BaseModel):
    track_acceptance_rate: bool = True
    log_level: str = "INFO"
    print_token_text: bool = True


class AppConfig(BaseModel):
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    speculative: SpeculativeConfig = Field(default_factory=SpeculativeConfig)
    remote_server: RemoteServerConfig = Field(default_factory=RemoteServerConfig)
    local_proxy: LocalProxyConfig = Field(default_factory=LocalProxyConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @classmethod
    def load_from_yaml(cls, path: Optional[str] = None) -> "AppConfig":
        if path is None:
            config_dir = Path(__file__).resolve().parent
            path = config_dir / "config.yaml"
        else:
            path = Path(path)

        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Allow environment variable overrides
        if "SPECROUTE_REMOTE_URL" in os.environ:
            data.setdefault("remote_server", {})["base_url"] = os.environ["SPECROUTE_REMOTE_URL"]
        if "SPECROUTE_BACKEND" in os.environ:
            data.setdefault("models", {})["local_engine_backend"] = os.environ["SPECROUTE_BACKEND"]
        if "SPECROUTE_LOOKAHEAD_K" in os.environ:
            data.setdefault("speculative", {})["lookahead_k"] = int(os.environ["SPECROUTE_LOOKAHEAD_K"])

        return cls(**data)


# Global singleton instance
config = AppConfig.load_from_yaml()

