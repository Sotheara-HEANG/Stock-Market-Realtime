"""
Quick test for Finnhub integration.

Usage:
    python test_realtime.py
"""

from dotenv import load_dotenv
load_dotenv()

from etl.extract import extract_stock_prices

if __name__ == "__main__":
    print("Fetching Finnhub stock price data...")
    df = extract_stock_prices(symbols=["AAPL", "MSFT", "NVDA"], history_years=2)

    if len(df) > 0:
        print(f"\nSuccess! Got {len(df)} data points")
        print("\nSample data:")
        print(df.head(10))
    else:
        print("\nNo data returned. Check that FINNHUB_API_KEY is set in .env")
