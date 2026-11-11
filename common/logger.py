"""Logging utilities for SpecRoute with specialized speculative decoding formatters."""

import logging
import sys
from typing import Optional


class SpeculativeFormatter(logging.Formatter):
    """Custom formatter with ANSI color coding and timestamp formatting."""

    GREY = "\x1b[38;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    BLUE = "\x1b[34;20m"
    CYAN = "\x1b[36;20m"
    BOLD_GREEN = "\x1b[32;1m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: GREY + "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s" + RESET,
        logging.INFO: CYAN + "[%(asctime)s]" + RESET + " %(message)s",
        logging.WARNING: YELLOW + "[%(asctime)s] [WARN] %(message)s" + RESET,
        logging.ERROR: BOLD_RED + "[%(asctime)s] [ERROR] %(message)s" + RESET,
        logging.CRITICAL: BOLD_RED + "[%(asctime)s] [CRITICAL] %(message)s" + RESET,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


def get_logger(name: str = "SpecRoute", level: str = "INFO") -> logging.Logger:
    """Configures and returns a logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(SpeculativeFormatter())
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.propagate = False
    return logger


class SpeculativeMetrics:
    """Tracks token acceptance statistics across the entire generation process."""

    def __init__(self):
        self.total_rounds: int = 0
        self.total_drafted: int = 0
        self.total_accepted: int = 0
        self.total_emitted: int = 0  # accepted + corrective + initial
        self.round_history: list = []

    def record_step(
        self,
        step: int,
        drafted: int,
        accepted: int,
        corrective_token_text: Optional[str] = None,
        latency_ms: float = 0.0,
    ) -> str:
        self.total_rounds += 1
        self.total_drafted += drafted
        self.total_accepted += accepted
        # Each round produces accepted tokens + 1 corrective token
        self.total_emitted += (accepted + 1)

        rate = (accepted / drafted * 100.0) if drafted > 0 else 0.0
        cum_rate = (self.total_accepted / self.total_drafted * 100.0) if self.total_drafted > 0 else 0.0

        corr_display = f" | Corrective: {repr(corrective_token_text)}" if corrective_token_text is not None else ""
        lat_display = f" | Latency: {latency_ms:.1f}ms" if latency_ms > 0 else ""

        step_log = (
            f"[Speculation] Step {step}: {accepted}/{drafted} tokens accepted "
            f"({rate:.1f}%){corr_display} | Cumulative: {self.total_accepted}/{self.total_drafted} "
            f"({cum_rate:.1f}%){lat_display}"
        )
        self.round_history.append({
            "step": step,
            "drafted": drafted,
            "accepted": accepted,
            "rate": rate,
            "latency_ms": latency_ms,
        })
        return step_log

    @property
    def cumulative_acceptance_rate(self) -> float:
        return (self.total_accepted / self.total_drafted) if self.total_drafted > 0 else 0.0

    @property
    def estimated_speedup(self) -> float:
        # Standard speculative decoding theoretical speedup approximation
        # S = (1 + alpha * K) / (1 + c) where c is draft cost ratio
        # Here effective tokens per target forward pass = (total_accepted + total_rounds) / total_rounds
        if self.total_rounds == 0:
            return 1.0
        return (self.total_accepted + self.total_rounds) / self.total_rounds

    def summary(self) -> str:
        cum_rate = self.cumulative_acceptance_rate * 100.0
        return (
            f"\n"
            f"======================= Speculative Decoding Summary =======================\n"
            f"  Total Speculative Rounds    : {self.total_rounds}\n"
            f"  Total Draft Tokens Guessed  : {self.total_drafted}\n"
            f"  Total Draft Tokens Accepted : {self.total_accepted}\n"
            f"  Total Tokens Generated      : {self.total_emitted}\n"
            f"  Overall Token Acceptance    : {self.total_accepted}/{self.total_drafted} ({cum_rate:.1f}%)\n"
            f"  Effective Tokens / Target Pass: {self.estimated_speedup:.2f}x\n"
            f"============================================================================"
        )

