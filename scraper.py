import requests
from bs4 import BeautifulSoup
import pandas as pd
import webbrowser


header = {
    "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

def scrape_data(header):
    url = 'https://finance.yahoo.com/screener/predefined/sec-ind_sec-largest-equities_technology/?offset=0&count=100'
    r = requests.get(url, headers=header)
    soup = BeautifulSoup(r.text, 'html.parser')

    symbols = []
    names = []
    prices = []
    changes = []
    percent_changes = []
    volumes = []
    market_caps = []
    pe_ratios = []

    for items in soup.find_all('tr', attrs={'class':'simpTblRow Bgc($hoverBgColor):h BdB Bdbc($seperatorColor) Bdbc($tableBorderBlue):h H(32px) Bgc($lv2BgColor)' }):
        for symbol in items.find_all('td', attrs={'aria-label':'Symbol'}):
            symbols.append(symbol.text)
        for name in items.find_all('td', attrs={'aria-label':'Name'}):
            names.append(name.text)
        for price in items.find_all('td', attrs={'aria-label':'Price (Intraday)'}):
            prices.append(price.text)
        for change in items.find_all('td', attrs={'aria-label':'Change'}):
            changes.append(change.text)
        for percent_change in items.find_all('td', attrs={'aria-label':'% Change'}):
            percent_changes.append(percent_change.text)
        for volume in items.find_all('td', attrs={'aria-label':'Volume'}):
            volumes.append(volume.text)
        for market_cap in items.find_all('td', attrs={'aria-label':'Market Cap'}):
            market_caps.append(market_cap.text)
        for pe_ratio in items.find_all('td', attrs={'aria-label':'PE Ratio (TTM)'}):
            pe_ratios.append(pe_ratio.text)

    data = {
        'Symbol' : symbols,
        'Name' : names,
        'Price' : prices,
        'Change' : changes,
        '%_Change' : percent_changes,
        'Volume' : volumes,
        'Market_Cap' : market_caps,
        'PE_Ratio' : pe_ratios
    }

    df = pd.DataFrame(data)
    df.to_csv("Technology-Sector-Data.csv")
    
def scrape_historical_data(header, symbol):
    url = 'https://finance.yahoo.com/quote/'+ symbol + '/history'
    r = requests.get(url, headers=header)
    soup = BeautifulSoup(r.text, 'html.parser')

    a = soup.find('a', attrs={'data-testid':'download-link'}).get('href')
    print(a)
    webbrowser.open(a)

scrape_historical_data(header, 'MSFT')    
    
    


