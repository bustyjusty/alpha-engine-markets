"""Shared number formatting.

Lives outside the service packages because both the MKR report and the
security-first pages print the same prices, and a level quoted two ways on two
screens is the kind of inconsistency that erodes trust in the whole dashboard.
"""

from __future__ import annotations

from typing import Callable

#: Currency codes to the symbol traders actually write.
CURRENCY_SYMBOLS = {
    "USD": "$", "EUR": "€", "GBP": "£", "GBp": "GBp ", "JPY": "¥",
    "CNY": "¥", "HKD": "HK$", "AUD": "A$", "CAD": "C$", "CHF": "CHF ",
    "SEK": "SEK ", "KRW": "₩", "TWD": "NT$", "INR": "₹", "SGD": "S$",
}


def money_formatter(
    reference: float, currency: str | None = None
) -> Callable[[float | None], str]:
    """Pick a decimal precision and a currency mark, then stick to both.

    A 4-unit stock and a 4,000-point index need different precision; choosing
    once per report keeps every level comparable. The currency comes from the
    security's own reference data — stamping "$" on an Amsterdam listing is the
    kind of error that reads as a rounding difference until it costs money. When
    the currency is unknown the numbers are printed bare rather than guessed.
    """
    decimals = 4 if reference < 1 else 3 if reference < 10 else 2
    mark = "" if currency is None else CURRENCY_SYMBOLS.get(currency, f"{currency} ")

    def render(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{mark}{value:,.{decimals}f}"

    return render


def percent(value: float | None, decimals: int = 1, signed: bool = False) -> str:
    """A ratio (0.153) as a percentage string ("15.3%"), or "n/a"."""
    if value is None:
        return "n/a"
    return f"{value * 100:{'+' if signed else ''}.{decimals}f}%"


def compact_number(value: float | None, currency: str | None = None) -> str:
    """Large figures as 1.42T / 68.3B / 940M, the way a terminal prints them."""
    if value is None:
        return "n/a"
    mark = "" if currency is None else CURRENCY_SYMBOLS.get(currency, f"{currency} ")
    magnitude = abs(value)
    for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= size:
            return f"{mark}{value / size:,.2f}{suffix}"
    return f"{mark}{value:,.0f}"
