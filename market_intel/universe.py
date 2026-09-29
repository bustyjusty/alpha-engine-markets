"""Cross-asset instrument universe for the global market recap.

Static reference data (not fetched), in the spirit of :mod:`market_intel.etf_catalog`:
the fixed grid of instruments a macro trader reads every morning, tagged by
**region** and **asset class** so the recap can be sliced either way.

Two modelling decisions carry most of the weight here:

* **Quote kind.** A move means different things per instrument. Equity indices
  move in percent; government bonds move in *basis points of yield*; FX moves in
  percent but its sign is only meaningful once you know which currency is the
  base. :class:`QuoteKind` records which, so the analysis layer can format and
  rank each block correctly instead of pretending everything is a percentage.

* **Proxies are labelled, not hidden.** Yahoo Finance publishes live yields for
  US Treasuries only. For every other rates market the universe falls back to a
  listed bond **ETF** and marks it ``is_proxy=True`` with ``inverse_yield=True``
  (an ETF price rally means yields fell). The UI surfaces that caveat rather
  than quietly presenting a total-return price as if it were a yield.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Region(str, Enum):
    """Trading region, ordered as the session clock runs."""

    APAC = "APAC"
    UK = "UK"
    EUROPE = "Europe"
    US = "US"
    GLOBAL = "Global"


class AssetClass(str, Enum):
    """Top-level asset class used to group a recap block."""

    EQUITIES = "Equities"
    FIXED_INCOME = "Fixed Income"
    CURRENCIES = "Currencies"
    COMMODITIES = "Commodities"


class QuoteKind(str, Enum):
    """How an instrument's quote should be read and its move expressed.

    Attributes:
        PRICE: An index level or futures price; move expressed in percent.
        YIELD: A yield quoted in percent; move expressed in basis points.
        FX: An exchange rate; move in percent, direction depends on the pair.
    """

    PRICE = "price"
    YIELD = "yield"
    FX = "fx"


@dataclass(frozen=True, slots=True)
class Instrument:
    """One tracked instrument.

    Attributes:
        symbol: Provider ticker (yfinance convention).
        label: Human-readable name shown in the UI.
        region: Trading region this instrument belongs to.
        asset_class: Block it is grouped under.
        quote: How to interpret and format its move.
        is_proxy: True when this stands in for something not directly quoted
            (e.g. a gilt ETF standing in for UK 10-year yields).
        inverse_yield: True when a rising price implies *falling* yields, so the
            UI can annotate the direction. Only meaningful for bond proxies.
        note: Short caveat or convention reminder shown beside the row.
    """

    symbol: str
    label: str
    region: Region
    asset_class: AssetClass
    quote: QuoteKind = QuoteKind.PRICE
    is_proxy: bool = False
    inverse_yield: bool = False
    note: str | None = None


def _eq(symbol: str, label: str, region: Region, note: str | None = None) -> Instrument:
    return Instrument(symbol, label, region, AssetClass.EQUITIES, note=note)


def _fx(symbol: str, label: str, region: Region, note: str | None = None) -> Instrument:
    return Instrument(
        symbol, label, region, AssetClass.CURRENCIES, QuoteKind.FX, note=note
    )


def _cmdty(symbol: str, label: str, region: Region = Region.GLOBAL) -> Instrument:
    return Instrument(symbol, label, region, AssetClass.COMMODITIES)


def _yld(symbol: str, label: str, region: Region) -> Instrument:
    return Instrument(symbol, label, region, AssetClass.FIXED_INCOME, QuoteKind.YIELD)


def _bond_proxy(symbol: str, label: str, region: Region, note: str) -> Instrument:
    return Instrument(
        symbol,
        label,
        region,
        AssetClass.FIXED_INCOME,
        QuoteKind.PRICE,
        is_proxy=True,
        inverse_yield=True,
        note=note,
    )


def _fred_series(symbol: str, label: str, region: Region) -> Instrument:
    """A spread or rate published only by FRED, quoted in percent."""
    return Instrument(
        symbol,
        label,
        region,
        AssetClass.FIXED_INCOME,
        QuoteKind.YIELD,
        note="FRED series - publishes next business day, so it lags the tape",
    )


_PRICE_UP_YIELDS_DOWN = "ETF proxy - price up means yields down"

#: Every instrument the recap tracks. Ordered for display within each block.
RECAP_UNIVERSE: tuple[Instrument, ...] = (
    # --- APAC equities --------------------------------------------------------
    _eq("^N225", "Nikkei 225", Region.APAC),
    _eq("^HSI", "Hang Seng", Region.APAC),
    _eq("^HSCE", "HS China Enterprises", Region.APAC),
    _eq("000001.SS", "Shanghai Composite", Region.APAC),
    _eq("399001.SZ", "Shenzhen Component", Region.APAC),
    _eq("^KS11", "KOSPI", Region.APAC),
    _eq("^TWII", "TAIEX", Region.APAC),
    _eq("^AXJO", "ASX 200", Region.APAC),
    _eq("^NSEI", "Nifty 50", Region.APAC),
    _eq("^BSESN", "Sensex", Region.APAC),
    _eq("^STI", "Straits Times", Region.APAC),
    _eq("^JKSE", "Jakarta Composite", Region.APAC),
    _eq("^KLSE", "FTSE Bursa Malaysia", Region.APAC),
    # --- UK equities ----------------------------------------------------------
    _eq("^FTSE", "FTSE 100", Region.UK, "Large-cap, mostly overseas earnings"),
    _eq("^FTMC", "FTSE 250", Region.UK, "The domestic UK read"),
    # --- Europe equities ------------------------------------------------------
    _eq("^STOXX", "STOXX Europe 600", Region.EUROPE),
    _eq("^STOXX50E", "Euro STOXX 50", Region.EUROPE),
    _eq("^GDAXI", "DAX", Region.EUROPE),
    _eq("^FCHI", "CAC 40", Region.EUROPE),
    _eq("FTSEMIB.MI", "FTSE MIB", Region.EUROPE),
    _eq("^IBEX", "IBEX 35", Region.EUROPE),
    _eq("^AEX", "AEX", Region.EUROPE),
    _eq("^SSMI", "SMI", Region.EUROPE),
    # --- US equities ----------------------------------------------------------
    _eq("^GSPC", "S&P 500", Region.US),
    _eq("^SPXEW", "S&P 500 Equal Weight", Region.US, "Breadth check vs cap-weighted"),
    _eq("^NDX", "Nasdaq 100", Region.US),
    _eq("^IXIC", "Nasdaq Composite", Region.US),
    _eq("^DJI", "Dow Jones Industrial", Region.US),
    _eq("^RUT", "Russell 2000", Region.US),
    _eq("^VIX", "VIX", Region.US, "Implied vol, not a return series"),
    # --- US sectors (drives the biggest-movers view) --------------------------
    _eq("XLK", "Technology (XLK)", Region.US),
    _eq("XLF", "Financials (XLF)", Region.US),
    _eq("XLE", "Energy (XLE)", Region.US),
    _eq("XLV", "Health Care (XLV)", Region.US),
    _eq("XLI", "Industrials (XLI)", Region.US),
    _eq("XLY", "Cons Discretionary (XLY)", Region.US),
    _eq("XLP", "Cons Staples (XLP)", Region.US),
    _eq("XLU", "Utilities (XLU)", Region.US),
    _eq("XLB", "Materials (XLB)", Region.US),
    _eq("XLRE", "Real Estate (XLRE)", Region.US),
    _eq("XLC", "Communication Svcs (XLC)", Region.US),
    _eq("SMH", "Semiconductors (SMH)", Region.US),
    # --- Fixed income: US Treasuries are the only true live yields ------------
    _yld("^IRX", "US 13-week bill", Region.US),
    _yld("^FVX", "US 5-year", Region.US),
    _yld("^TNX", "US 10-year", Region.US),
    _yld("^TYX", "US 30-year", Region.US),
    _bond_proxy("TLT", "US 20y+ Treasuries (TLT)", Region.US, _PRICE_UP_YIELDS_DOWN),
    _bond_proxy("IEF", "US 7-10y Treasuries (IEF)", Region.US, _PRICE_UP_YIELDS_DOWN),
    _bond_proxy("SHY", "US 1-3y Treasuries (SHY)", Region.US, _PRICE_UP_YIELDS_DOWN),
    _bond_proxy("LQD", "US IG credit (LQD)", Region.US, "ETF proxy - credit sentiment"),
    _bond_proxy("HYG", "US high yield (HYG)", Region.US, "ETF proxy - risk appetite tell"),
    _bond_proxy("EMB", "EM sovereign USD (EMB)", Region.GLOBAL, "ETF proxy - EM risk premium"),
    _bond_proxy("IGLT.L", "UK gilts (IGLT.L)", Region.UK, "ETF proxy - no live gilt yield on this feed"),
    _bond_proxy("IBTS.L", "UK short gilts (IBTS.L)", Region.UK, "ETF proxy - front-end BoE pricing"),
    _bond_proxy("SEGA.L", "EUR govt bonds (SEGA.L)", Region.EUROPE, "ETF proxy - bunds/OATs/BTPs blended"),
    _bond_proxy("IEAC.L", "EUR IG credit (IEAC.L)", Region.EUROPE, "ETF proxy"),
    _bond_proxy("1482.T", "JGB 7-10y (1482.T)", Region.APAC, _PRICE_UP_YIELDS_DOWN),
    _bond_proxy("IAF.AX", "AU composite bond (IAF.AX)", Region.APAC, "ETF proxy"),
    # FRED-only macro series. No Yahoo ticker exists for any of these, and they
    # are the first thing a macro desk reads: credit risk premium, the shape of
    # the curve, and market-implied inflation. FRED publishes next business day,
    # so these lag the live tape by roughly one session.
    _fred_series("FRED:HY_OAS", "US high-yield OAS", Region.US),
    _fred_series("FRED:IG_OAS", "US IG corporate OAS", Region.US),
    _fred_series("FRED:10Y2Y", "US 10y-2y curve", Region.US),
    _fred_series("FRED:BREAKEVEN10", "US 10y breakeven inflation", Region.US),
    # --- Currencies -----------------------------------------------------------
    _fx("DX-Y.NYB", "Dollar Index (DXY)", Region.US, "Up = broad USD strength"),
    _fx("USDJPY=X", "USD/JPY", Region.APAC, "Up = yen weaker"),
    _fx("CNY=X", "USD/CNY", Region.APAC, "Up = yuan weaker"),
    _fx("AUDUSD=X", "AUD/USD", Region.APAC, "Up = aussie stronger"),
    _fx("NZDUSD=X", "NZD/USD", Region.APAC, "Up = kiwi stronger"),
    _fx("USDKRW=X", "USD/KRW", Region.APAC, "Up = won weaker"),
    _fx("USDSGD=X", "USD/SGD", Region.APAC, "Up = SGD weaker"),
    _fx("USDTWD=X", "USD/TWD", Region.APAC, "Up = TWD weaker"),
    _fx("USDINR=X", "USD/INR", Region.APAC, "Up = rupee weaker"),
    _fx("USDHKD=X", "USD/HKD", Region.APAC, "Pegged band 7.75-7.85"),
    _fx("GBPUSD=X", "GBP/USD", Region.UK, "Up = sterling stronger"),
    _fx("EURGBP=X", "EUR/GBP", Region.UK, "Up = sterling weaker vs euro"),
    _fx("EURUSD=X", "EUR/USD", Region.EUROPE, "Up = euro stronger"),
    _fx("USDCHF=X", "USD/CHF", Region.EUROPE, "Up = franc weaker"),
    _fx("EURCHF=X", "EUR/CHF", Region.EUROPE, "SNB watch level"),
    _fx("USDCAD=X", "USD/CAD", Region.US, "Up = loonie weaker"),
    _fx("USDMXN=X", "USD/MXN", Region.US, "Carry / risk barometer"),
    # --- Commodities ----------------------------------------------------------
    _cmdty("GC=F", "Gold"),
    _cmdty("SI=F", "Silver"),
    _cmdty("PL=F", "Platinum"),
    _cmdty("PA=F", "Palladium"),
    _cmdty("HG=F", "Copper"),
    _cmdty("KC=F", "Coffee"),
    _cmdty("CT=F", "Cotton"),
    _cmdty("BZ=F", "Brent crude", Region.EUROPE),
    _cmdty("CL=F", "WTI crude", Region.US),
    _cmdty("NG=F", "Henry Hub nat gas", Region.US),
    _cmdty("ZW=F", "Wheat", Region.US),
    _cmdty("ZC=F", "Corn", Region.US),
    _cmdty("ZS=F", "Soybeans", Region.US),
    _cmdty("LE=F", "Live cattle", Region.US),
)

#: Display order for region tabs - session clock order, APAC first.
REGION_ORDER: tuple[Region, ...] = (
    Region.APAC,
    Region.UK,
    Region.EUROPE,
    Region.US,
    Region.GLOBAL,
)

#: Display order for asset-class blocks within a region.
ASSET_CLASS_ORDER: tuple[AssetClass, ...] = (
    AssetClass.EQUITIES,
    AssetClass.FIXED_INCOME,
    AssetClass.CURRENCIES,
    AssetClass.COMMODITIES,
)


def instruments_for(
    region: Region | None = None, asset_class: AssetClass | None = None
) -> list[Instrument]:
    """Return universe members matching the given filters, in display order.

    Args:
        region: Restrict to one region; ``None`` means every region.
        asset_class: Restrict to one asset class; ``None`` means all.
    """
    return [
        instrument
        for instrument in RECAP_UNIVERSE
        if (region is None or instrument.region is region)
        and (asset_class is None or instrument.asset_class is asset_class)
    ]


def symbols_for(
    region: Region | None = None, asset_class: AssetClass | None = None
) -> list[str]:
    """Return just the ticker symbols matching the given filters."""
    return [instrument.symbol for instrument in instruments_for(region, asset_class)]


def find(symbol: str) -> Instrument | None:
    """Look up an instrument by symbol (case-insensitive), or None."""
    target = symbol.strip().upper()
    for instrument in RECAP_UNIVERSE:
        if instrument.symbol.upper() == target:
            return instrument
    return None
