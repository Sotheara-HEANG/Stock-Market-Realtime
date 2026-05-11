"""
Quick test for real-time finance API integration.

Usage:
    python test_realtime.py
"""

from dotenv import load_dotenv
load_dotenv()

from etl.extract import extract_realtime_finance

if __name__ == "__main__":
    print("Fetching real-time finance data...")
    df = extract_realtime_finance(symbols=["AAPL", "GOOGL", "BTC-USD"])

    if len(df) > 0:
        print(f"\n✓ Success! Got {len(df)} data points")
        print("\nSample data:")
        print(df.head(10))
    else:
        print("\n✗ No data returned. Check that RAPIDAPI_KEY is set in .env")
