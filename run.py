"""SpecRoute entry point - starts the local API server with the speculative orchestrator."""

import logging
import sys

import uvicorn
import yaml

from local_engine.api_server import app, init_orchestrator


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    log_level = config.get("logging", {}).get("level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    init_orchestrator(config)

    host = config["api_server"]["host"]
    port = config["api_server"]["port"]

    logging.getLogger("specroute").info(
        "Starting SpecRoute API server on %s:%d", host, port
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
