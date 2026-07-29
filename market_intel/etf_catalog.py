"""Curated ETF reference catalog, grouped by investment theme.

Static reference data (not fetched): a starting universe for the ETF
Explorer so users can browse by theme instead of typing tickers. Symbols
are all liquid US-listed ETFs supported by the yfinance provider.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CatalogEtf:
    """One catalog entry: ticker plus a human-readable name."""

    symbol: str
    name: str


ETF_CATALOG: dict[str, list[CatalogEtf]] = {
    "Broad market": [
        CatalogEtf("SPY", "SPDR S&P 500"),
        CatalogEtf("QQQ", "Invesco Nasdaq-100"),
        CatalogEtf("DIA", "SPDR Dow Jones Industrial Average"),
        CatalogEtf("IWM", "iShares Russell 2000"),
        CatalogEtf("RSP", "Invesco S&P 500 Equal Weight"),
    ],
    "US sectors": [
        CatalogEtf("XLK", "Technology Select Sector SPDR"),
        CatalogEtf("XLF", "Financial Select Sector SPDR"),
        CatalogEtf("XLE", "Energy Select Sector SPDR"),
        CatalogEtf("XLV", "Health Care Select Sector SPDR"),
        CatalogEtf("XLI", "Industrial Select Sector SPDR"),
        CatalogEtf("XLY", "Consumer Discretionary Select SPDR"),
        CatalogEtf("XLP", "Consumer Staples Select Sector SPDR"),
        CatalogEtf("XLU", "Utilities Select Sector SPDR"),
        CatalogEtf("XLB", "Materials Select Sector SPDR"),
        CatalogEtf("XLRE", "Real Estate Select Sector SPDR"),
        CatalogEtf("XLC", "Communication Services Select SPDR"),
    ],
    "Semiconductors & AI": [
        CatalogEtf("SMH", "VanEck Semiconductor"),
        CatalogEtf("SOXX", "iShares Semiconductor"),
        CatalogEtf("IGV", "iShares Expanded Tech-Software"),
        CatalogEtf("BOTZ", "Global X Robotics & AI"),
        CatalogEtf("AIQ", "Global X Artificial Intelligence & Tech"),
    ],
    "Thematic & industry": [
        CatalogEtf("XBI", "SPDR S&P Biotech"),
        CatalogEtf("ITA", "iShares US Aerospace & Defense"),
        CatalogEtf("JETS", "US Global Jets"),
        CatalogEtf("TAN", "Invesco Solar"),
        CatalogEtf("LIT", "Global X Lithium & Battery Tech"),
        CatalogEtf("URA", "Global X Uranium"),
        CatalogEtf("HACK", "Amplify Cybersecurity"),
        CatalogEtf("ARKK", "ARK Innovation"),
    ],
    "International": [
        CatalogEtf("EFA", "iShares MSCI EAFE (developed ex-US)"),
        CatalogEtf("EEM", "iShares MSCI Emerging Markets"),
        CatalogEtf("FXI", "iShares China Large-Cap"),
        CatalogEtf("KWEB", "KraneShares CSI China Internet"),
        CatalogEtf("EWJ", "iShares MSCI Japan"),
        CatalogEtf("INDA", "iShares MSCI India"),
    ],
    "Rates, credit & commodities": [
        CatalogEtf("TLT", "iShares 20+ Year Treasury Bond"),
        CatalogEtf("IEF", "iShares 7-10 Year Treasury Bond"),
        CatalogEtf("LQD", "iShares Investment Grade Corporate Bond"),
        CatalogEtf("HYG", "iShares High Yield Corporate Bond"),
        CatalogEtf("GLD", "SPDR Gold Shares"),
        CatalogEtf("SLV", "iShares Silver Trust"),
        CatalogEtf("USO", "United States Oil Fund"),
    ],
}
"""Theme name -> ETFs. Ordering is display order in the Explorer."""


def themes() -> list[str]:
    """All catalog theme names in display order."""
    return list(ETF_CATALOG)


def etfs_for_theme(theme: str) -> list[CatalogEtf]:
    """Catalog entries for a theme (empty list for unknown themes)."""
    return list(ETF_CATALOG.get(theme, []))


def lookup(symbol: str) -> CatalogEtf | None:
    """Find a catalog entry by ticker, if the ETF is catalogued."""
    wanted = symbol.strip().upper()
    for entries in ETF_CATALOG.values():
        for entry in entries:
            if entry.symbol == wanted:
                return entry
    return None


def all_symbols() -> list[str]:
    """Every catalogued ticker (unique, display order)."""
    seen: list[str] = []
    for entries in ETF_CATALOG.values():
        for entry in entries:
            if entry.symbol not in seen:
                seen.append(entry.symbol)
    return seen
