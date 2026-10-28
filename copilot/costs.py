"""Turn token counts into money.

Prices are per million tokens and come from environment variables, because they differ by
provider and change over time. They default to 0, which is right for a local Ollama model.

    export PRICE_INPUT_PER_MTOK=0.25 PRICE_OUTPUT_PER_MTOK=1.25
"""
import os


def prices():
    return (float(os.getenv("PRICE_INPUT_PER_MTOK", "0")), float(os.getenv("PRICE_OUTPUT_PER_MTOK", "0")))


def usd(prompt_tokens, completion_tokens):
    price_in, price_out = prices()
    return round(prompt_tokens / 1e6 * price_in + completion_tokens / 1e6 * price_out, 6)
