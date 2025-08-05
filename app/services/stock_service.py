import yfinance as yf
from typing import List, Dict, Any
import time
import random
from datetime import datetime

def get_nifty_500_stocks():
    """Get optimized stock data from Yahoo Finance - focused on top Indian stocks"""
    try:
        return get_top_indian_stocks()
    except Exception as e:
        print(f"Yahoo Finance method failed: {e}")
        return {
            "status": "error",
            "message": f"Failed to fetch stock data from Yahoo Finance: {str(e)}",
            "error": "Real-time data unavailable",
            "count": 0,
            "stocks": []
        }

def get_top_indian_stocks():
    """Get a randomized selection of Indian stocks from Yahoo Finance with variety each time"""
    
    # Expanded list of 200+ Indian stocks across all sectors for variety
    all_stocks = [
        # Top 30 by market cap (higher probability)
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
        "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
        "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "NESTLEIND.NS",
        "ULTRACEMCO.NS", "TITAN.NS", "WIPRO.NS", "BAJFINANCE.NS", "SUNPHARMA.NS",
        "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS", "INDUSINDBK.NS", "ICICIGI.NS",
        "HDFCAMC.NS", "POWERGRID.NS", "COALINDIA.NS", "NTPC.NS", "ONGC.NS",
        
        # Banking & Financial Services (40+ stocks)
        "PFC.NS", "RECLTD.NS", "CHOLAFIN.NS", "MUTHOOTFIN.NS", "SHRIRAMFIN.NS",
        "MAZDOCK.NS", "IDFC.NS", "BANKINDIA.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS",
        "CANBK.NS", "PNB.NS", "UNIONBANK.NS", "BANKBARODA.NS", "INDIANB.NS",
        "RBLBANK.NS", "YESBANK.NS", "EQUITASBNK.NS", "UJJIVAN.NS", "AUBANK.NS",
        "DCBBANK.NS", "SOUTHBANK.NS", "IOB.NS", "CENTRALBK.NS", "PSB.NS",
        "J&KBANK.NS", "KARURBANK.NS", "TMGBANK.NS", "CITYUNION.NS", "DHANLAXMI.NS",
        "NAINITAL.NS", "UJJIVANSFB.NS", "ESAFSFB.NS", "JANABANK.NS", "FINPIPE.NS",
        "MANAPPURAM.NS", "AAVAS.NS", "GRUH.NS", "REPCO.NS", "CAPF.NS",
        
        # Information Technology (30+ stocks)  
        "HCLTECH.NS", "TECHM.NS", "MINDTREE.NS", "MPHASIS.NS", "PERSISTENT.NS", 
        "LTTS.NS", "OFSS.NS", "COFORGE.NS", "HEXAWARE.NS", "CYIENT.NS",
        "RUPA.NS", "POLYCAB.NS", "KPITTECH.NS", "ZENSAR.NS", "SONATSOFTW.NS",
        "NIITTECH.NS", "INTELLECT.NS", "RAMCOCEM.NS", "MASTEK.NS", "ROUTE.NS",
        "NEWGEN.NS", "BIRLASOFT.NS", "DATAPATTNS.NS", "RAJESHEXPO.NS", "TANLA.NS",
        "BHARATGEAR.NS", "SUBEXLTD.NS", "MINDACORP.NS", "BARTRONICS.NS", "AXISCADES.NS",
        
        # Energy & Utilities (25+ stocks)
        "IOC.NS", "BPCL.NS", "HINDPETRO.NS", "GAIL.NS", "TATAPOWER.NS", 
        "ADANIPORTS.NS", "ADANIENT.NS", "ADANIGREEN.NS", "ADANITRANS.NS", "NHPC.NS",
        "SJVN.NS", "THERMAX.NS", "BHEL.NS", "CESC.NS", "TORNTPOWER.NS",
        "JSW.NS", "RELCAPITAL.NS", "GUJGAS.NS", "IGL.NS", "MGL.NS",
        "PETRONET.NS", "GSPL.NS", "AEGISCHEM.NS", "DEEPAKNI.NS", "KIRIINDUS.NS",
        
        
        # Consumer Goods & FMCG (30+ stocks)
        "BRITANNIA.NS", "DABUR.NS", "MARICO.NS", "GODREJCP.NS", "COLPAL.NS",
        "TATACONSUM.NS", "UBL.NS", "VBL.NS", "EMAMILTD.NS", "JYOTHYLAB.NS",
        "PGHH.NS", "GILLETTE.NS", "HONAUT.NS", "VGUARD.NS", "BATAINDIA.NS",
        "RELAXO.NS", "PAGEIND.NS", "ADVENZYMES.NS", "FINEORG.NS", "ZYDUSWELL.NS",
        "ABBOTINDIA.NS", "SANOFI.NS", "GLAXO.NS", "PFIZER.NS", "NOVARTIS.NS",
        "3MINDIA.NS", "CASTROLIND.NS", "SCHAEFFLER.NS", "SKFINDIA.NS", "TIMKEN.NS",
        
        # Automotive (25+ stocks)
        "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS",
        "ASHOKLEY.NS", "TVSMOTOR.NS", "ESCORTS.NS", "MRF.NS", "APOLLOTYRE.NS",
        "BALKRISIND.NS", "CEATLTD.NS", "JK.NS", "MOTHERSUMI.NS", "BOSCHLTD.NS",
        "MAHINDCIE.NS", "BHARATFORG.NS", "RAMKRISHNA.NS", "MINDAIND.NS", "ENDURANCE.NS",
        "SUBROS.NS", "LUMAX.NS", "FIEM.NS", "GABRIEL.NS", "TIINDIA.NS",
        
        # Metals & Mining (20+ stocks)
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "SAIL.NS",
        "JINDALSTEL.NS", "NMDC.NS", "MOIL.NS", "HINDZINC.NS", "NATIONALUM.NS",
        "COALINDIA.NS", "WELCORP.NS", "WELSPUNIND.NS", "JSPL.NS", "KALYANI.NS",
        "RATNAMANI.NS", "GRAVITA.NS", "APLAPOLLO.NS", "JINDALPOLY.NS", "ORIENTCEM.NS",
        
        # Pharmaceuticals (25+ stocks)
        "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "BIOCON.NS", "TORNTPHARM.NS", 
        "AUROPHARMA.NS", "LUPIN.NS", "ALKEM.NS", "GLENMARK.NS", "CADILAHC.NS",
        "STRIDES.NS", "LALPATHLAB.NS", "THYROCARE.NS", "METROPOLIS.NS", "KRSNAA.NS",
        "SEQUENT.NS", "AAVAS.NS", "SUVEN.NS", "NATCOPHAR.NS", "GRANULES.NS",
        "AJANTPHARM.NS", "DIVIS.NS", "REDDY.NS", "MANKIND.NS", "ERIS.NS",
        
        # Cement & Construction (15+ stocks)
        "SHREECEM.NS", "GRASIM.NS", "ACC.NS", "AMBUJACEM.NS", "JKCEMENT.NS",
        "HEIDELBERG.NS", "ORIENTCEM.NS", "CENTURYTEX.NS", "RAMCOCEM.NS", "PRISMCEM.NS",
        "INDIACEM.NS", "J&KBANK.NS", "KESORAMIND.NS", "STARCEMENT.NS", "SANGAMIND.NS",
        
        # Telecom & Media (15+ stocks)
        "IDEA.NS", "ZEEL.NS", "SUNTV.NS", "NETWORK18.NS", "TVTODAY.NS",
        "HATHWAY.NS", "DISHTTV.NS", "GTLINFRA.NS", "RCOM.NS", "TTML.NS",
        "SAREGAMA.NS", "BALAJITELE.NS", "PURAVANKARA.NS", "ADLABS.NS", "ENIL.NS",
        
        # Textiles & Apparel (20+ stocks)  
        "RAYMOND.NS", "ARVIND.NS", "RUPA.NS", "PAGEIND.NS", "WELSPUNIND.NS",
        "VARDHMAN.NS", "TRIDENT.NS", "ALOKTEXT.NS", "BANSWRAS.NS", "CENTEX.NS",
        "GARFIBRES.NS", "HIMATSEIDE.NS", "INDORAMA.NS", "JAICORPLTD.NS", "KPR.NS",
        "KPRMILL.NS", "LAXMIMACH.NS", "NITIN.NS", "PATSPIN.NS", "SPENTEX.NS",
        
        # Agriculture & Food Processing (15+ stocks)
        "RALLIS.NS", "UPL.NS", "COROMANDEL.NS", "INSECTICID.NS", "ZUARI.NS",
        "MADRASFERT.NS", "GSFC.NS", "NFL.NS", "RCF.NS", "FACT.NS",
        "KRIBHCO.NS", "SHREYAS.NS", "PATANJALI.NS", "DAAWAT.NS", "VSTIND.NS"
    ]
    
    # Remove delisted stocks that cause errors
    delisted_stocks = ["LTI.NS", "NIITTECH.NS", "CADILAHC.NS", "DALMIACEM.NS", "RCOM.NS", "ALOKTEXT.NS"]
    all_stocks = [stock for stock in all_stocks if stock not in delisted_stocks]
    
    # **ROTATION LOGIC**: Select different stocks each time
    # Create seed based on current date for daily rotation
    today_seed = int(datetime.now().strftime("%Y%m%d"))
    random.seed(today_seed)
    
    # Select 60-80 stocks randomly (varies each day)
    min_stocks = 60
    max_stocks = min(80, len(all_stocks))
    num_stocks = random.randint(min_stocks, max_stocks)
    
    # Ensure we always include some top performers (first 30 stocks)
    guaranteed_stocks = all_stocks[:30]  # Top 30 market cap stocks
    remaining_stocks = all_stocks[30:]   # Rest of the stocks
    
    # Randomly select additional stocks
    additional_needed = num_stocks - len(guaranteed_stocks)
    if additional_needed > 0:
        additional_stocks = random.sample(remaining_stocks, min(additional_needed, len(remaining_stocks)))
        selected_stocks = guaranteed_stocks + additional_stocks
    else:
        selected_stocks = guaranteed_stocks[:num_stocks]
    
    # Shuffle the final list for variety
    random.shuffle(selected_stocks)
    
    print(f"📊 Daily Stock Selection: {len(selected_stocks)} stocks (Seed: {today_seed})")
    print(f"🔄 Rotation ensures different stocks each day!")
    
    print(f"Fetching data for {len(selected_stocks)} selected Indian stocks...")
    start_time = time.time()
    
    stocks = []
    
    # Process sequentially with delays to avoid rate limiting
    def fetch_single_stock(symbol, retry_count=0):
        try:
            ticker = yf.Ticker(symbol)
            # Get basic info with timeout
            info = ticker.get_info()
            hist = ticker.history(period="2d")
            
            if hist.empty:
                return None
                
            current_price = float(hist['Close'].iloc[-1])
            prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
            change = current_price - prev_close
            pchange = (change / prev_close) * 100 if prev_close > 0 else 0
            
            return {
                "symbol": symbol.replace(".NS", ""),
                "companyName": info.get("longName", info.get("shortName", symbol.replace(".NS", ""))),
                "industry": info.get("industry", info.get("sector", "Unknown")),
                "lastPrice": round(current_price, 2),
                "dayHigh": round(float(hist['High'].iloc[-1]), 2),
                "dayLow": round(float(hist['Low'].iloc[-1]), 2),
                "previousClose": round(prev_close, 2),
                "change": round(change, 2),
                "pChange": round(pchange, 2),
                "marketCap": info.get("marketCap", 0),
                "volume": int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0,
                "source": "Yahoo Finance"
            }
            
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "too many requests" in error_msg:
                if retry_count < 2:  # Retry up to 2 times for rate limiting
                    print(f"Rate limited for {symbol}, retrying in 3 seconds...")
                    time.sleep(3)
                    return fetch_single_stock(symbol, retry_count + 1)
            print(f"Error fetching {symbol}: {e}")
            return None
    
    # Use very conservative approach to avoid API rate limits - sequential processing
    stocks = []
    failed_stocks = []
    
    for i, symbol in enumerate(selected_stocks):
        try:
            # Add delay between each request to avoid rate limiting
            if i > 0:
                time.sleep(0.5)  # 500ms delay between requests
            
            result = fetch_single_stock(symbol)
            if result:
                stocks.append(result)
            else:
                failed_stocks.append(symbol)
                
            if (i + 1) % 15 == 0:
                print(f"Progress: {len(stocks)} successful, {len(failed_stocks)} failed out of {i + 1} attempted (Total planned: {len(selected_stocks)})")
                
        except Exception as e:
            print(f"Failed to process {symbol}: {e}")
            failed_stocks.append(symbol)
    
    end_time = time.time()
    print(f"Successfully fetched {len(stocks)} stocks in {end_time - start_time:.2f} seconds")
    if failed_stocks:
        print(f"Failed to fetch {len(failed_stocks)} stocks: {failed_stocks[:5]}{'...' if len(failed_stocks) > 5 else ''}")
    return stocks

def get_all_nse_stocks():
    """Alias for get_nifty_500_stocks for backward compatibility"""
    return get_nifty_500_stocks()
