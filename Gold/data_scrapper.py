#FULL AI GENERATED CODE (yes I m not proud of myself)

import yfinance as yf
import json
from datetime import datetime

def generate_synthetic_data():
    print("=" * 60)
    print("FETCHING REAL GOLD DATA FROM YAHOO FINANCE")
    print("=" * 60)
    
    gold = yf.download('GC=F', start='1972-01-01', end=datetime.now().strftime('%Y-%m-%d'))
    
    dates = gold.index.strftime('%Y-%m-%d').tolist()
    prices = [float(p) for p in gold['Close'].values]
    
    print()
    print("=" * 60)
    print("✅ FETCH SUCCESSFUL!")
    print("=" * 60)
    print(f"Data Summary:")
    print(f"  Start Date: {dates[0]}")
    print(f"  End Date:   {dates[-1]}")
    print(f"  Total Data: {len(prices)} points")
    print(f"  Max Price:  ${max(prices):.2f}")
    print(f"  Current Price: ${prices[-1]:.2f}")
    print()
    
    data = {
        'dates': dates,
        'prices': prices,
        'ticker': 'GC=F',
        'description': 'Gold Futures Historical Data',
        'source': 'Yahoo Finance',
        'fetched_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    filename = 'real_db.json'
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"✅ Saved to '{filename}'")

if __name__ == "__main__":
    generate_synthetic_data()