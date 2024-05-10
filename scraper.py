import requests
from bs4 import BeautifulSoup
import pandas as pd
import webbrowser

#Write your USER-AGENT header into this dictionary
header = {
    'User-Agent' : 'Your user-agent'
}
#Tu jest funkcja, która odpowiada za scrapowanie danych bieżących z yahoo finance
def scrape_data(header):
    url = 'https://finance.yahoo.com/screener/predefined/sec-ind_sec-largest-equities_technology/?offset=0&count=100' #Tu chyba widzisz, że przypisuje adres url do zmiennej
    r = requests.get(url, headers=header) #Tutaj wysyłamy requesta zeby połaczyc sie ze strona z linku url
    soup = BeautifulSoup(r.text, 'html.parser') #r.text zawiera cały html z tej strony, w drugim parametrze wybieramy parser, jest ich kilka ale bieremy html.parser
    #w zmiennej soup bedzie cały html juz sparsowany tak, że można na nim jakies operacje robic, cos wyszukac itp.

    symbols = []
    names = []
    prices = []
    changes = []
    percent_changes = []
    volumes = []
    market_caps = []
    pe_ratios = []
     #Na tej stronie yahoo finance jest jebitna tabela, w której są zawarte te wszystkie dane co ich potzebujemy. Tu jest pętla w której przechodzimy po każdej komorce w kazdym wierszu.
    for items in soup.find_all('tr', attrs={'class':'simpTblRow Bgc($hoverBgColor):h BdB Bdbc($seperatorColor) Bdbc($tableBorderBlue):h H(32px) Bgc($lv2BgColor)' }): #tr to jest obiekt jednego wiersza w tabeli, nazwa klasy jest akurat pojebana na tej stronie i nwm co znaczy XD
        for symbol in items.find_all('td', attrs={'aria-label':'Symbol'}): #Potem przegladamy komórki, za pomocą atrybuty 'aria-label' w html jestesmy w stanie rozróżnić jaka komórka przechowuje jakie dane np. tutaj jest Symbol
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
            
#Robimy słownik z tych wszyskich list co były zadeklarowane powyżej pętli
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
    #Robimy data frame z tego słownika i zapisyjemy w pliku .csv, jeżeli taki plik już jest to chyba powinien się nadpisać wsm nie sprawdzalem xd
    df = pd.DataFrame(data)
    df.to_csv("Technology-Sector-Data.csv")
    
def scrape_historical_data(header, symbol):
    url = 'https://finance.yahoo.com/quote/'+ symbol + '/history'
    r = requests.get(url, headers=header)
    soup = BeautifulSoup(r.text, 'html.parser')
    #Tu początek taki sam jak w poprzedniej funkcji

    a = soup.find('a', attrs={'data-testid':'download-link'}).get('href')
    print(a) #Tu w zmiennej a zapisuemy link z pobieraniem pliku .csv z danymi historycznymi. Znajdujemy go przez atrybut 'data-testid'
    webbrowser.open(a) #otwieramy link w przeglądarce, plik pobrany.
    #I właśnie tu sie chuj zapisuje w pobranych normalnie a trzeba by było wykminić jak to zrobić żeby się zapisał w najlepiej w tym forlderze co mamy w nim cały projekt
    #Chociaż wsm to nie ma aż takiego znaczenia ale chyba byłoby tak wygodniej

 
    
    
