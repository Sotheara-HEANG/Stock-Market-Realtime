"""
enrich.py — derived features added to the cleaned wide Spark DataFrame.

Three enrichments, applied in order by enrich():

  1. add_gdp_yoy()           — year-over-year GDP growth calculated from gdp_usd_bn levels
                               using a Spark lag() window (distinct from IMF's reported
                               gdp_growth_pct, which is used as a cross-check)
  2. add_governance_composite() — simple average of the 6 WGI indicators per row,
                               only calculated when ≥ 3 of the 6 are non-null
  3. add_regional_averages() — joins a continent lookup, then computes per-(continent, year)
                               averages for gdp_growth_pct and governance_composite

Usage:
    from etl.enrich import enrich
    enriched_df = enrich(wide_df, spark)
"""

from __future__ import annotations

import warnings

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Year-over-year GDP growth (calculated from GDP levels)
# ---------------------------------------------------------------------------

def add_gdp_yoy(wide_df: DataFrame) -> DataFrame:
    """
    Derive gdp_growth_yoy_calc = (gdp_usd_bn[t] - gdp_usd_bn[t-1]) / gdp_usd_bn[t-1] * 100

    Partitioned by country_code so the lag doesn't bleed across countries.
    Output column: gdp_growth_yoy_calc (float, %)
    Rows where gdp_usd_bn is null in either year produce null — not imputed.
    """
    w = Window.partitionBy("country_code").orderBy("year")
    gdp_prev = F.lag("gdp_usd_bn").over(w)

    df = wide_df.withColumn(
        "gdp_growth_yoy_calc",
        F.when(
            gdp_prev.isNotNull() & (gdp_prev != 0),
            F.round((F.col("gdp_usd_bn") - gdp_prev) / gdp_prev * 100, 4),
        ).otherwise(F.lit(None).cast("double")),
    )

    non_null = df.filter(F.col("gdp_growth_yoy_calc").isNotNull()).count()
    print(f"  gdp_growth_yoy_calc: {non_null:,} non-null values")
    return df


# ---------------------------------------------------------------------------
# 2. Governance composite score (mean of ≥ 3 WGI indicators)
# ---------------------------------------------------------------------------

_WGI_COLS = [
    "control_of_corruption",
    "government_effectiveness",
    "political_stability",
    "regulatory_quality",
    "rule_of_law",
    "voice_and_accountability",
]

_MIN_WGI = 3   # require at least this many non-null WGI columns to compute composite


def add_governance_composite(wide_df: DataFrame) -> DataFrame:
    """
    governance_composite = mean of available WGI indicators per row.

    All 6 WGI columns are scored ~-2.5 to +2.5 (standardised), so a simple
    mean is comparable across rows with different numbers of available columns.
    Set to null when fewer than _MIN_WGI columns are present.
    """
    available = [c for c in _WGI_COLS if c in wide_df.columns]

    # Count non-null WGI values per row
    non_null_count = sum(
        F.when(F.col(c).isNotNull(), F.lit(1)).otherwise(F.lit(0))
        for c in available
    )
    # Sum of non-null WGI values per row
    wgi_sum = sum(F.coalesce(F.col(c), F.lit(0.0)) for c in available)

    df = wide_df.withColumn("_wgi_count", non_null_count).withColumn(
        "governance_composite",
        F.when(
            F.col("_wgi_count") >= _MIN_WGI,
            F.round(wgi_sum / F.col("_wgi_count"), 4),
        ).otherwise(F.lit(None).cast("double")),
    ).drop("_wgi_count")

    non_null = df.filter(F.col("governance_composite").isNotNull()).count()
    print(f"  governance_composite: {non_null:,} non-null values  (from {len(available)} WGI cols, min {_MIN_WGI} required)")
    return df


# ---------------------------------------------------------------------------
# 3. Regional averages by continent
# ---------------------------------------------------------------------------

# ISO-3 → continent mapping (covers all countries present in the dataset)
_CONTINENT_MAP: dict[str, str] = {
    # Africa
    "AGO": "Africa", "BDI": "Africa", "BEN": "Africa", "BFA": "Africa",
    "BWA": "Africa", "CAF": "Africa", "CIV": "Africa", "CMR": "Africa",
    "COD": "Africa", "COG": "Africa", "COM": "Africa", "CPV": "Africa",
    "DJI": "Africa", "DZA": "Africa", "EGY": "Africa", "ERI": "Africa",
    "ETH": "Africa", "GAB": "Africa", "GHA": "Africa", "GIN": "Africa",
    "GMB": "Africa", "GNB": "Africa", "GNQ": "Africa", "KEN": "Africa",
    "LBR": "Africa", "LBY": "Africa", "LSO": "Africa", "MAR": "Africa",
    "MDG": "Africa", "MLI": "Africa", "MOZ": "Africa", "MRT": "Africa",
    "MUS": "Africa", "MWI": "Africa", "NAM": "Africa", "NER": "Africa",
    "NGA": "Africa", "RWA": "Africa", "SDN": "Africa", "SEN": "Africa",
    "SLE": "Africa", "SOM": "Africa", "SSD": "Africa", "STP": "Africa",
    "SWZ": "Africa", "SYC": "Africa", "TCD": "Africa", "TGO": "Africa",
    "TUN": "Africa", "TZA": "Africa", "UGA": "Africa", "ZAF": "Africa",
    "ZMB": "Africa", "ZWE": "Africa",
    # Asia  (ISO-3 + Polity5 scodes)
    "AFG": "Asia", "ARM": "Asia", "AZE": "Asia", "BGD": "Asia",
    "BHR": "Asia", "BRN": "Asia", "BTN": "Asia", "CHN": "Asia",
    "CYP": "Asia", "GEO": "Asia", "HKG": "Asia", "IDN": "Asia",
    "IND": "Asia", "IRN": "Asia", "IRQ": "Asia", "ISR": "Asia",
    "JOR": "Asia", "JPN": "Asia", "KAZ": "Asia", "KGZ": "Asia",
    "KHM": "Asia", "KOR": "Asia", "KWT": "Asia", "LAO": "Asia",
    "LBN": "Asia", "LKA": "Asia", "MAC": "Asia", "MDV": "Asia",
    "MMR": "Asia", "MNG": "Asia", "MYS": "Asia", "NPL": "Asia",
    "OMN": "Asia", "PAK": "Asia", "PHL": "Asia", "PRK": "Asia",
    "PSE": "Asia", "QAT": "Asia", "SAU": "Asia", "SGP": "Asia",
    "SYR": "Asia", "TJK": "Asia", "TKM": "Asia", "TLS": "Asia",
    "THA": "Asia", "TUR": "Asia", "TWN": "Asia", "UZB": "Asia",
    "VNM": "Asia", "YEM": "Asia", "ARE": "Asia",
    # Polity5 scodes — Asia
    "AFQ": "Asia",                          # Afghanistan (Polity5 variant)
    "AZQ": "Asia",                          # Azerbaijan
    "BHU": "Asia",                          # Bhutan
    "BNG": "Asia",                          # Bangladesh
    "CAM": "Asia",                          # Cambodia
    "CHN": "Asia",                          # China
    "INS": "Asia",                          # Indonesia
    "IRE": "Asia",                          # not Ireland — Polity5 uses IRE for unassigned
    "KUW": "Asia",                          # Kuwait
    "KYR": "Asia",                          # Kyrgyzstan
    "KZK": "Asia",                          # Kazakhstan
    "LAT": "Europe",                        # Latvia (Polity5 LAT)
    "LEB": "Asia",                          # Lebanon
    "MAL": "Asia",                          # Malaysia
    "MAS": "Asia",                          # Malaysia alternate
    "MLD": "Asia",                          # Maldives
    "MYA": "Asia",                          # Myanmar
    "NEP": "Asia",                          # Nepal
    "OMA": "Asia",                          # Oman
    "PAP": "Oceania",                       # Papua New Guinea (Polity5)
    "PHI": "Asia",                          # Philippines
    "PKS": "Asia",                          # Pakistan
    "ROK": "Asia",                          # Republic of Korea
    "RVN": "Asia",                          # Vietnam (South, historical)
    "SAR": "Asia",                          # Saudi Arabia
    "SIN": "Asia",                          # Singapore
    "SRI": "Asia",                          # Sri Lanka
    "TAJ": "Asia",                          # Tajikistan
    "TAW": "Asia",                          # Taiwan
    "TAZ": "Asia",                          # Turkmenistan
    "THA": "Asia",                          # Thailand
    "THI": "Asia",                          # Thailand (Polity5 variant)
    "UAE": "Asia",                          # UAE (Polity5 scode)
    "UZB": "Asia",                          # Uzbekistan (already in ISO list)
    "VIE": "Asia",                          # Vietnam (Polity5)
    "WBG": "Asia",                          # West Bank/Gaza
    # Europe  (ISO-3 + Polity5 scodes)
    "ALB": "Europe", "AND": "Europe", "AUT": "Europe", "BEL": "Europe",
    "BGR": "Europe", "BIH": "Europe", "BLR": "Europe", "CHE": "Europe",
    "CZE": "Europe", "DEU": "Europe", "DNK": "Europe", "ESP": "Europe",
    "EST": "Europe", "FIN": "Europe", "FRA": "Europe", "GBR": "Europe",
    "GRC": "Europe", "HRV": "Europe", "HUN": "Europe", "IRL": "Europe",
    "ISL": "Europe", "ITA": "Europe", "LIE": "Europe", "LTU": "Europe",
    "LUX": "Europe", "LVA": "Europe", "MCO": "Europe", "MDA": "Europe",
    "MKD": "Europe", "MLT": "Europe", "MNE": "Europe", "NLD": "Europe",
    "NOR": "Europe", "POL": "Europe", "PRT": "Europe", "ROU": "Europe",
    "RUS": "Europe", "SMR": "Europe", "SRB": "Europe", "SVK": "Europe",
    "SVN": "Europe", "SWE": "Europe", "UKR": "Europe", "XKX": "Europe",
    # Polity5 scodes — Europe
    "ALG": "Africa",                        # Algeria (misclassified as Europe above — actually Africa)
    "BAD": "Europe",                        # Baden (historical German state)
    "BAV": "Europe",                        # Bavaria (historical)
    "BOS": "Europe",                        # Bosnia
    "BUL": "Europe",                        # Bulgaria (Polity5)
    "CRO": "Europe",                        # Croatia
    "CZR": "Europe",                        # Czechoslovakia
    "DEN": "Europe",                        # Denmark
    "FRN": "Europe",                        # France
    "GDR": "Europe",                        # East Germany (historical)
    "GFR": "Europe",                        # West Germany (historical)
    "GMY": "Europe",                        # Germany / Prussia (Polity5)
    "GRG": "Asia",                          # Georgia (Polity5)
    "HON": "North America",                 # Honduras (Polity5)
    "KOS": "Europe",                        # Kosovo
    "LIT": "Europe",                        # Lithuania
    "MNT": "Europe",                        # Montenegro
    "MOD": "Europe",                        # Moldova
    "MON": "Europe",                        # Monaco
    "NTH": "Europe",                        # Netherlands
    "POR": "Europe",                        # Portugal
    "RUM": "Europe",                        # Romania
    "SAX": "Europe",                        # Saxony (historical)
    "SER": "Europe",                        # Serbia
    "SIC": "Europe",                        # Sicily (historical)
    "SLO": "Europe",                        # Slovenia/Slovakia
    "SPN": "Europe",                        # Spain
    "SWD": "Europe",                        # Sweden
    "TUS": "Europe",                        # Tuscany (historical)
    "UKG": "Europe",                        # United Kingdom
    "UVK": "Europe",                        # Kosovo (Polity5)
    "VAT": "Europe",                        # Vatican
    "YGS": "Europe",                        # Yugoslavia (Polity5)
    "YUG": "Europe",                        # Yugoslavia (ISO variant)
    # North America  (ISO-3 + Polity5 scodes)
    "ATG": "North America", "BHS": "North America", "BLZ": "North America",
    "BRB": "North America", "CAN": "North America", "CRI": "North America",
    "CUB": "North America", "DMA": "North America", "DOM": "North America",
    "GRD": "North America", "GTM": "North America", "HND": "North America",
    "HTI": "North America", "JAM": "North America", "KNA": "North America",
    "LCA": "North America", "MEX": "North America", "NIC": "North America",
    "PAN": "North America", "SLV": "North America", "TTO": "North America",
    "USA": "North America", "VCT": "North America",
    "ABW": "North America", "BMU": "North America", "CYM": "North America",
    "GRL": "North America", "PRI": "North America", "VIR": "North America",
    # Polity5 scodes — North America / Caribbean
    "COS": "North America",                 # Costa Rica
    "GUA": "North America",                 # Guatemala
    "HAI": "North America",                 # Haiti
    "SAL": "North America",                 # El Salvador
    # South America  (ISO-3 + Polity5 scodes)
    "ARG": "South America", "BOL": "South America", "BRA": "South America",
    "CHL": "South America", "COL": "South America", "ECU": "South America",
    "GUY": "South America", "PER": "South America", "PRY": "South America",
    "SUR": "South America", "URY": "South America", "VEN": "South America",
    # Polity5 scodes — South America
    "PAR": "South America",                 # Paraguay
    "URU": "South America",                 # Uruguay
    # Africa  (Polity5 scodes)
    "ANG": "Africa",                        # Angola
    "BFO": "Africa",                        # Burkina Faso
    "BOT": "Africa",                        # Botswana
    "BUI": "Africa",                        # Burundi
    "CAO": "Africa",                        # Cameroon
    "CAP": "Africa",                        # Cape Verde
    "CEN": "Africa",                        # Central African Republic
    "CHA": "Africa",                        # Chad
    "CON": "Africa",                        # Congo (Polity5)
    "ETI": "Africa",                        # Eritrea
    "ETM": "Africa",                        # East Timor (misclassified — actually Asia)
    "EQG": "Africa",                        # Equatorial Guinea
    "GAM": "Africa",                        # Gambia
    "GUI": "Africa",                        # Guinea
    "IVO": "Africa",                        # Côte d'Ivoire
    "KEN": "Africa",                        # Kenya (already in ISO)
    "LES": "Africa",                        # Lesotho
    "LIB": "Africa",                        # Liberia
    "MAA": "Africa",                        # Mauritania
    "MAG": "Africa",                        # Madagascar
    "MAW": "Africa",                        # Malawi
    "MLD": "Africa",                        # Mali (Polity5 MLD vs ISO MLI)
    "MOR": "Africa",                        # Morocco
    "MZM": "Africa",                        # Mozambique
    "NIG": "Africa",                        # Niger / Nigeria
    "NIR": "Africa",                        # Nigeria (alternate)
    "OFS": "Africa",                        # Orange Free State (historical)
    "RWA": "Africa",                        # Rwanda (already ISO)
    "SAF": "Africa",                        # South Africa
    "SIE": "Africa",                        # Sierra Leone
    "SOL": "Africa",                        # Somalia
    "SSA": "Africa",                        # South Sudan
    "SSU": "Africa",                        # South Sudan (alternate)
    "SUD": "Africa",                        # Sudan
    "SWA": "Africa",                        # Eswatini/Swaziland
    "TOG": "Africa",                        # Togo
    "TRI": "North America",                 # Trinidad (Polity5)
    "TUN": "Africa",                        # Tunisia (already ISO)
    "UPC": "Africa",                        # Upper Volta / Burkina Faso
    "ZAI": "Africa",                        # Zaire (DRC, historical)
    "ZAM": "Africa",                        # Zambia
    "ZIM": "Africa",                        # Zimbabwe
    # Oceania  (ISO-3 + Polity5 scodes)
    "AUS": "Oceania", "FJI": "Oceania", "FSM": "Oceania", "KIR": "Oceania",
    "MHL": "Oceania", "NRU": "Oceania", "NZL": "Oceania", "PLW": "Oceania",
    "PNG": "Oceania", "SLB": "Oceania", "TON": "Oceania", "TUV": "Oceania",
    "VUT": "Oceania", "WSM": "Oceania",
    "ASM": "Oceania", "GUM": "Oceania",
    "AUL": "Oceania",                       # Australia (Polity5)
    "NEW": "Oceania",                       # New Zealand (Polity5)
    "SOL": "Oceania",                       # Solomon Islands (Polity5)
    # Remaining historical / alternate Polity5 scodes
    "BAH": "Asia",                          # Bahrain (Polity5 BAH vs ISO BHR)
    "DRV": "Asia",                          # Democratic Republic of Vietnam (historical)
    "EDE": "Europe",                        # East Germany (alternate Polity5 code)
    "GCL": "South America",                 # Gran Colombia (historical)
    "MAE": "Africa",                        # Mauritania (alternate)
    "PMA": "North America",                 # Panama (Polity5)
    "USR": "Europe",                        # USSR (historical)
    "WRT": "Europe",                        # Württemberg (historical German state)
    "YAR": "Asia",                          # Yemen Arab Republic (historical)
    "YPR": "Asia",                          # Yemen People's Democratic Republic (historical)
    # Q-suffix codes are WGI/World Bank regional aggregates (not individual countries)
    # APQ, CAQ, CBQ, CMQ, EAQ, EEQ, EUQ, MEQ, NAQ, NMQ, OAE, PIQ,
    # SAQ, SEQ, SMQ, SSQ, WEQ, WHQ — intentionally left unmapped
}

_REGIONAL_AVG_COLS = ["gdp_growth_pct", "governance_composite"]


def add_regional_averages(wide_df: DataFrame, spark: SparkSession) -> DataFrame:
    """
    Join a continent lookup and add per-(continent, year) averages for
    gdp_growth_pct and governance_composite.

    Output columns:
        continent                   — string, from _CONTINENT_MAP
        regional_avg_gdp_growth     — avg gdp_growth_pct by (continent, year)
        regional_avg_governance     — avg governance_composite by (continent, year)
    """
    # Build continent lookup as a small Spark DF
    continent_rows = [(code, continent) for code, continent in _CONTINENT_MAP.items()]
    continent_df = spark.createDataFrame(continent_rows, ["country_code", "continent"])

    df = wide_df.join(continent_df, on="country_code", how="left")

    # Calculate per-(continent, year) averages only over rows that have a value
    for col_name in _REGIONAL_AVG_COLS:
        if col_name not in df.columns:
            continue
        alias = f"regional_avg_{'gdp_growth' if 'gdp' in col_name else 'governance'}"
        regional = (
            df.filter(F.col(col_name).isNotNull() & F.col("continent").isNotNull())
            .groupBy("continent", "year")
            .agg(F.round(F.avg(col_name), 4).alias(alias))
        )
        df = df.join(regional, on=["continent", "year"], how="left")

    covered = df.filter(F.col("continent").isNotNull()).count()
    print(f"  continent mapped: {covered:,} rows assigned  ({df.count() - covered:,} unmapped)")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def enrich(wide_df: DataFrame, spark: SparkSession) -> DataFrame:
    """
    Apply all three enrichments to the cleaned wide DataFrame.

    Args:
        wide_df : cleaned wide Spark DF from transform.transform()
        spark   : active SparkSession

    Returns:
        Enriched Spark DataFrame with additional columns:
            gdp_growth_yoy_calc, governance_composite,
            continent, regional_avg_gdp_growth, regional_avg_governance
    """
    print("=== Enrich ===")

    print("  [1/3] YoY GDP growth from GDP levels...")
    df = add_gdp_yoy(wide_df)

    print("  [2/3] Governance composite score...")
    df = add_governance_composite(df)

    print("  [3/3] Regional averages by continent...")
    df = add_regional_averages(df, spark)

    new_cols = ["gdp_growth_yoy_calc", "governance_composite",
                "continent", "regional_avg_gdp_growth", "regional_avg_governance"]
    print(f"\n  Added columns: {new_cols}")
    print(f"  Final shape: {df.count():,} rows  ×  {len(df.columns)} columns")
    return df
