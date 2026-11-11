#!/usr/bin/env python
"""SpecRoute Target Server Launcher.

Launches the remote target verification server hosting the larger model
(e.g., Qwen2.5-7B-Instruct or Llama-3-8B).
"""

import os
import argparse
import uvicorn
from config.settings import config
from common.logger import get_logger

logger = get_logger("TargetLauncher")


def main():
    parser = argparse.ArgumentParser(description="SpecRoute Target Verification Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8001, help="Bind port")
    parser.add_argument("--model", type=str, default=config.models.target_model_id, help="Target Model ID")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode for rapid testing")
    args = parser.parse_args()

    if args.simulate:
        os.environ["SPECROUTE_SIMULATE_TARGET"] = "true"

    logger.info("=" * 70)
    logger.info("  SpecRoute: Remote Target Verification Server")
    logger.info(f"  Target Model ID    : {args.model}")
    logger.info(f"  Simulation Mode    : {args.simulate}")
    logger.info(f"  Verification API   : http://{args.host}:{args.port}/v1/speculative/verify")
    logger.info(f"  WebSocket Channel  : ws://{args.host}:{args.port}/v1/speculative/ws")
    logger.info("=" * 70)

    uvicorn.run("server.target_server:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

