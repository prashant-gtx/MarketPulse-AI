import yfinance as yf
from datetime import datetime
import concurrent.futures
import time
from live_scraper import get_live_price

# In-memory cache for market data
MARKET_CACHE = None
LAST_MARKET_FETCH = 0
MARKET_CACHE_TTL = 10 # 10 seconds cache for overall market data

def get_market_data():
    """
    Fetches live (delayed) data for Nifty 50, Sensex, and commodities in parallel.
    Uses in-memory caching for 60 seconds.
    """
    global MARKET_CACHE, LAST_MARKET_FETCH
    
    if MARKET_CACHE and (time.time() - LAST_MARKET_FETCH < MARKET_CACHE_TTL):
        return MARKET_CACHE

    indices_tickers = {
        "Nifty 50": "^NSEI",
        "Sensex": "^BSESN"
    }
    
    commodities_tickers = {
        "Gold": "GC=F",
        "Silver": "SI=F",
        "USD/INR": "INR=X",
        "EUR/INR": "EURINR=X",
        "BTC/USD": "BTC-USD"
    }
    
    indices_data = {}
    commodities_data = {}
    
    def fetch_index(name, symbol):
        try:
            ticker = yf.Ticker(symbol)
            # Fetch 1-day history with 5-minute intervals for sparkline
            hist = ticker.history(period="1d", interval="5m")
            
            # Fallback if 1d 5m is unavailable (e.g. market closed recently or data gap)
            if hist.empty or len(hist) < 5:
                hist = ticker.history(period="5d", interval="15m")
            if hist.empty:
                hist = ticker.history(period="1mo")
            
            # FIX: Use fast_info for the most up-to-date 'current' price. 
            # 'history' can be stale (e.g. on weekends/holidays with special sessions).
            # TRY LIVE SCRAPER FIRST for current price
            # This bypasses yfinance staleness entirely
            live_price = get_live_price(symbol)
            
            if live_price:
                current = live_price
                # Try to get previous close from yfinance for change calc
                try:
                    prev_close = ticker.fast_info.previous_close
                except:
                    if not hist.empty:
                        prev_close = hist["Close"].iloc[-1] # fallback to last close
                    else:
                        prev_close = current # no change
            else:
                # FALLBACK to yfinance fast_info/history
                try:
                    current = ticker.fast_info.last_price
                    prev_close = ticker.fast_info.previous_close
                    if not current: raise ValueError("No fast_info price")
                except:
                    if hist.empty: return None
                    current = hist["Close"].iloc[-1]
                    try:
                        prev_close = ticker.fast_info.previous_close
                    except:
                        prev_close = hist["Open"].iloc[0]
            
            change = current - prev_close
            percent_change = (change / prev_close) * 100
            
            sparkline_points = []
            if not hist.empty:
                sparkline_points = hist["Close"].tail(20).tolist()
                
                # Append live point if it's new
                try:
                    if live_price or (hist.index[-1].date() < datetime.now().date()):
                         # Always append if we have a fresh live price
                         sparkline_points.append(current)
                except:
                    pass
            
            return name, {
                "price": round(current, 2),
                "change": round(change, 2),
                "percent_change": round(percent_change, 2),
                "symbol": symbol,
                "history": sparkline_points
            }
        except: return None

    def fetch_commodity(name, symbol):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            price = info.last_price
            prev_close = info.previous_close
            if price and prev_close:
                change = price - prev_close
                percent_change = (change / prev_close) * 100
                return name, {
                    "price": round(price, 2),
                    "change": round(change, 2),
                    "percent_change": round(percent_change, 2),
                    "symbol": symbol
                }
        except: return None

    # Fetch all in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
        idx_futures = [executor.submit(fetch_index, n, s) for n, s in indices_tickers.items()]
        cmd_futures = [executor.submit(fetch_commodity, n, s) for n, s in commodities_tickers.items()]
        
        for f in concurrent.futures.as_completed(idx_futures):
            res = f.result()
            if res: indices_data[res[0]] = res[1]
            
        for f in concurrent.futures.as_completed(cmd_futures):
            res = f.result()
            if res: commodities_data[res[0]] = res[1]

    result = {
        "indices": indices_data,
        "commodities": commodities_data
    }
    
    MARKET_CACHE = result
    LAST_MARKET_FETCH = time.time()
    return result

# Simple in-memory cache: {symbol: {"data": data, "timestamp": timestamp}}
STOCK_CACHE = {}

def get_stock_details(symbol):
    """
    Fetches detailed data for a specific stock.
    Results are cached for 30 seconds to prevent slow repeated fetching.
    """
    try:
        # Check Cache
        if symbol in STOCK_CACHE:
            cached = STOCK_CACHE[symbol]
            age = (datetime.now() - cached["timestamp"]).total_seconds()
            if age < 10: # 10 Seconds Cache
                # print(f"Serving {symbol} from cache")
                return cached["data"]

        # Append .NS for NSE stocks if not present (heuristic)
        # But let's assume valid symbol is passed or handle generically
        search_symbol = symbol
        if not symbol.endswith(".NS") and not symbol.endswith(".BO") and not "=X" in symbol and not "^" in symbol:
             # Assume NSE by default for Indian context if no extension
             search_symbol = f"{symbol}.NS"
        
        ticker = yf.Ticker(search_symbol)
        info = ticker.fast_info
        
        # history for day range fallback if fast_info missing
        # hist = ticker.history(period="1d") 
        
        # Essential Data
        # TRY LIVE SCRAPER FIRST
        live_price = get_live_price(search_symbol)
        
        if live_price:
            price = live_price
        else:
            price = info.last_price
            
        prev_close = info.previous_close
        open_price = info.open
        day_high = info.day_high
        day_low = info.day_low
        
        # Extended Data (may need ticker.info for some)
        # fast_info is faster but has less data. 
        # market_cap is in fast_info
        market_cap = info.market_cap
        
        # Volume usually requires history or regular info
        # Let's try to get recent volume
        volume = info.last_volume

        # 52 Week
        year_high = info.year_high
        year_low = info.year_low
        
        change = price - prev_close
        p_change = (change / prev_close) * 100
        
        result = {
            "symbol": symbol,
            "name": search_symbol, # Fallback, ideally get strict name
            "price": round(price, 2),
            "change": round(change, 2),
            "percent_change": round(p_change, 2),
            "previous_close": round(prev_close, 2),
            "open": round(open_price, 2),
            "day_high": round(day_high, 2),
            "day_low": round(day_low, 2),
            "year_high": round(year_high, 2),
            "year_low": round(year_low, 2),
            "market_cap": market_cap,
            "volume": volume,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Update Cache
        STOCK_CACHE[symbol] = {
            "data": result,
            "timestamp": datetime.now()
        }
        
        return result
    except Exception as e:
        print(f"Error fetching details for {symbol}: {e}")
        return None

def get_stock_history(symbol, period="1mo"):
    """
    Fetches historical data for a graph.
    Period options: 1d, 5d, 1mo, 6mo, 1y, 5y, max
    """
    try:
        # Resolve symbol
        search_symbol = symbol
        if not symbol.endswith(".NS") and not symbol.endswith(".BO") and not "=X" in symbol and not "^" in symbol:
             search_symbol = f"{symbol}.NS"
        
        ticker = yf.Ticker(search_symbol)
        
        # Determine interval based on period to optimize data points
        interval = "1d"
        if period == "1d":
            interval = "5m"
        elif period == "5d":
            interval = "15m"
        elif period == "1mo":
            interval = "1d" # or 90m if available
            
        hist = ticker.history(period=period, interval=interval)
        
        if hist.empty:
            return []
            
        # Format data for frontend (recharts)
        # Array of { date: "ISO string", price: 123.45 }
        data = []
        for index, row in hist.iterrows():
            # index is DatetimeIndex
            data.append({
                "date": index.isoformat(), 
                "price": round(row["Close"], 2),
                "high": round(row["High"], 2),
                "low": round(row["Low"], 2)
            })
            
        # FIX: Append live price if history is stale (e.g. Special Session or Weekend lag)
        # We try to get the live price from Google Finance scraper
        try:
            live_price = get_live_price(search_symbol)
            
            if not hist.empty:
                last_hist_time = hist.index[-1]
                
                # Logic: If the last history point is NOT from today, we append "Live" as a new point
                # This catches:
                # 1. Missing "Yesterday" (if today is Monday and yesterday was Sunday session)
                # 2. Today's live movement if yfinance hasn't updated yet
                
                today_date = datetime.now().date()
                last_hist_date = last_hist_time.date()
                
                if live_price and last_hist_date < today_date:
                     # Calculate a pseudo-timestamp. 
                     # If data is missing for yesterday, this point acts as "Now"
                     data.append({
                         "date": datetime.now().astimezone().isoformat(), 
                         "price": round(live_price, 2),
                         "high": round(live_price, 2), 
                         "low": round(live_price, 2)
                     })
                     print(f"Appended live fallback for {symbol}: {live_price}")
                     
        except Exception as ex:
            print(f"Error appending live point: {ex}")
            
        return data

    except Exception as e:
        print(f"Error fetching history for {symbol}: {e}")
        return []


def get_stock_financials(symbol):
    """
    Fetches quarterly financial data (Income Statement).
    """
    try:
        # Resolve symbol
        search_symbol = symbol
        if not symbol.endswith(".NS") and not symbol.endswith(".BO") and not "=X" in symbol and not "^" in symbol:
             search_symbol = f"{symbol}.NS"
        
        ticker = yf.Ticker(search_symbol)
        
        # Get Quarterly Income Statement
        # transposed so columns are dates
        fin = ticker.quarterly_income_stmt
        
        if fin.empty:
            return []
            
        # We want columns (dates) to be the list items
        # Structure: [ {date: '...', revenue: ...}, ... ]
        
        results = []
        # Loop through columns (dates)
        for date in fin.columns:
            try:
                col_data = fin[date]
                
                # Helper to safely get value by key (handling various naming conventions if needed)
                # yfinance keys are usually Standardized
                
                revenue = col_data.get("Total Revenue", 0)
                if not revenue or str(revenue) == 'nan':
                     revenue = col_data.get("Operating Revenue", 0)
                     
                net_income = col_data.get("Net Income", 0)
                ebitda = col_data.get("EBITDA", col_data.get("Normalized EBITDA", 0))
                eps = col_data.get("Basic EPS", 0)
                
                import math
                def sanitize(val):
                    if not val or (isinstance(val, float) and math.isnan(val)):
                        return 0
                    return float(val)

                results.append({
                    "date": date.strftime("%b %Y"), # e.g., Dec 2024
                    "revenue": sanitize(revenue),
                    "net_income": sanitize(net_income),
                    "ebitda": sanitize(ebitda),
                    "eps": sanitize(eps)
                })
            except Exception as e:
                print(f"Error parsing financial col {date}: {e}")
                continue
                
        # Return last 5 quarters
        return results[:5]

    except Exception as e:
        print(f"Error fetching financials for {symbol}: {e}")
        return []

if __name__ == "__main__":
    print(get_stock_financials("RELIANCE"))

