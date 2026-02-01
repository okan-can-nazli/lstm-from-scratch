"""
Real Gold Price Data Fetcher - Yahoo Finance Only
==================================================
This script fetches actual historical gold prices from Yahoo Finance
and saves them for use in the LSTM training.

Ticker: GC=F (Gold Futures)
"""

import numpy as np
import json
from datetime import datetime, timedelta

def fetch_gold_prices_yahoo(start_year=None, days=365):
    """
    Fetch real gold price data from Yahoo Finance
    
    Installation required:
    pip install yfinance
    
    Parameters:
    - start_year: if specified, fetch from this year to present (e.g., 2000)
    - days: number of days to fetch (if start_year not specified)
    """
    try:
        import yfinance as yf
        
        print("=" * 60)
        print("Fetching real gold price data from Yahoo Finance...")
        print("=" * 60)
        print()
        print(f"Ticker: GC=F (Gold Futures)")
        
        # Fetch data
        ticker = yf.Ticker("GC=F")
        
        # Get historical data
        end_date = datetime.now()
        
        if start_year:
            start_date = datetime(start_year, 1, 1)
            print(f"Period: From {start_year} to present")
        else:
            start_date = end_date - timedelta(days=days)
            print(f"Period: Last {days} days")
        
        print()
        print("Downloading... please wait...")
        print()
        
        hist = ticker.history(start=start_date, end=end_date)
        
        if hist.empty:
            print("⚠️  No data for GC=F. Trying alternative ticker GLD (Gold ETF)...")
            print()
            # Try SPDR Gold Shares ETF as backup
            ticker = yf.Ticker("GLD")
            hist = ticker.history(start=start_date, end=end_date)
        
        if hist.empty:
            print("❌ Failed to fetch data. Check internet connection.")
            print()
            return None
        
        # Extract closing prices
        prices = hist['Close'].values
        dates = hist.index.strftime('%Y-%m-%d').tolist()
        
        print("=" * 60)
        print("✅ DOWNLOAD SUCCESSFUL!")
        print("=" * 60)
        print()
        print(f"Data Summary:")
        print(f"  Total days: {len(prices)}")
        print(f"  Date range: {dates[0]} to {dates[-1]}")
        print(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        print(f"  Average price: ${prices.mean():.2f}")
        print()
        
        # Save to file
        data = {
            'dates': dates,
            'prices': prices.tolist(),
            'ticker': 'GC=F',
            'description': 'Gold Futures - Historical Closing Prices',
            'source': 'Yahoo Finance',
            'fetched_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('gold_prices.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        print("✅ Data saved to 'gold_prices.json'")
        print()
        
        # Show sample data
        print("Sample data (last 5 days):")
        for i in range(-5, 0):
            print(f"  {dates[i]}: ${prices[i]:.2f}")
        print()
        
        return prices, dates
        
    except ImportError:
        print("=" * 60)
        print("❌ ERROR: yfinance not installed!")
        print("=" * 60)
        print()
        print("To install, run:")
        print("  pip install yfinance")
        print()
        return None
    except Exception as e:
        print("=" * 60)
        print(f"❌ ERROR: {e}")
        print("=" * 60)
        print()
        return None


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("GOLD PRICE DATA FETCHER")
    print("=" * 60)
    print()
    
    print("What data do you want?")
    print()
    print("  1) ALL data from year 2000 to present (recommended)")
    print("  2) Last 365 days only")
    print("  3) Custom year range")
    print()
    
    choice = input("Choose option (1/2/3): ").strip()
    print()
    
    if choice == "1":
        # Fetch all data from 2000
        result = fetch_gold_prices_yahoo(start_year=2000)
        
    elif choice == "2":
        # Fetch last 365 days
        result = fetch_gold_prices_yahoo(days=365)
        
    elif choice == "3":
        # Custom year
        year = input("Start from which year? (e.g., 2010): ").strip()
        year = int(year) if year else 2000
        result = fetch_gold_prices_yahoo(start_year=year)
        
    else:
        print("❌ Invalid choice. Using default (2000 to present)")
        result = fetch_gold_prices_yahoo(start_year=2000)
    
    print()
    print("=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print()
    print("1. ✅ Data downloaded and saved to 'gold_prices.json'")
    print("2. 🚀 Now run: python advanced_task_real_data.py")
    print("3. 🎯 Your LSTM will train on REAL gold prices!")
    print()
    print("=" * 60)
