#!/usr/bin/env python
"""SpecRoute Local Proxy Launcher.

Launches the OpenAI-compatible proxy orchestrating client-side speculative
decoding and connecting to the remote target server.
"""

import sys
import argparse
import uvicorn
from config.settings import config
from common.logger import get_logger

logger = get_logger("Launcher")


def main():
    parser = argparse.ArgumentParser(description="SpecRoute Local Orchestrator Proxy")
    parser.add_argument("--host", type=str, default=config.local_proxy.host, help="Bind host")
    parser.add_argument("--port", type=int, default=config.local_proxy.port, help="Bind port")
    parser.add_argument("--remote-url", type=str, default=None, help="Remote server URL (Colab/Kaggle ngrok)")
    parser.add_argument("--backend", type=str, default=None, choices=["simulation", "huggingface", "llama_cpp", "ollama"], help="Draft engine backend")
    parser.add_argument("--lookahead-k", type=int, default=None, help="Lookahead window K")
    args = parser.parse_args()

    if args.remote_url:
        config.remote_server.base_url = args.remote_url
    if args.backend:
        config.models.local_engine_backend = args.backend
    if args.lookahead_k:
        config.speculative.lookahead_k = args.lookahead_k

    logger.info("=" * 70)
    logger.info("  SpecRoute: Hybrid Local-Cloud Speculative Decoding Proxy")
    logger.info(f"  Local Draft Model  : {config.models.draft_model_id} ({config.models.local_engine_backend})")
    logger.info(f"  Remote Target Host : {config.remote_server.base_url}")
    logger.info(f"  Lookahead Window K : {config.speculative.lookahead_k}")
    logger.info(f"  LibreChat Endpoint : http://{args.host}:{args.port}/v1")
    logger.info("=" * 70)

    uvicorn.run("proxy.main:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

