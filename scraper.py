import os.path

import requests
from bs4 import BeautifulSoup
import pandas as pd
import webbrowser

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
}


#Tu jest funkcja, która odpowiada za scrapowanie danych bieżących z yahoo finance
def scrape_data(header):
    url = 'https://finance.yahoo.com/screener/predefined/sec-ind_sec-largest-equities_technology/?offset=0&count=100'  #Tu chyba widzisz, że przypisuje adres url do zmiennej
    r = requests.get(url, headers=header)  #Tutaj wysyłamy requesta zeby połaczyc sie ze strona z linku url
    soup = BeautifulSoup(r.text,
                         'html.parser')  #r.text zawiera cały html z tej strony, w drugim parametrze wybieramy parser, jest ich kilka ale bieremy html.parser
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
    for items in soup.find_all('tr', attrs={
        'class': 'simpTblRow Bgc($hoverBgColor):h BdB Bdbc($seperatorColor) Bdbc($tableBorderBlue):h H(32px) Bgc($lv2BgColor)'}):  #tr to jest obiekt jednego wiersza w tabeli, nazwa klasy jest akurat pojebana na tej stronie i nwm co znaczy XD
        for symbol in items.find_all('td', attrs={
            'aria-label': 'Symbol'}):  #Potem przegladamy komórki, za pomocą atrybuty 'aria-label' w html jestesmy w stanie rozróżnić jaka komórka przechowuje jakie dane np. tutaj jest Symbol
            symbols.append(symbol.text)
        for name in items.find_all('td', attrs={'aria-label': 'Name'}):
            names.append(name.text)
        for price in items.find_all('td', attrs={'aria-label': 'Price (Intraday)'}):
            prices.append(price.text)
        for change in items.find_all('td', attrs={'aria-label': 'Change'}):
            changes.append(change.text)
        for percent_change in items.find_all('td', attrs={'aria-label': '% Change'}):
            percent_changes.append(percent_change.text)
        for volume in items.find_all('td', attrs={'aria-label': 'Volume'}):
            volumes.append(volume.text)
        for market_cap in items.find_all('td', attrs={'aria-label': 'Market Cap'}):
            market_caps.append(market_cap.text)
        for pe_ratio in items.find_all('td', attrs={'aria-label': 'PE Ratio (TTM)'}):
            pe_ratios.append(pe_ratio.text)

    #Robimy słownik z tych wszyskich list co były zadeklarowane powyżej pętli
    data = {
        'Symbol': symbols,
        'Name': names,
        'Price': prices,
        'Change': changes,
        '%_Change': percent_changes,
        'Volume': volumes,
        'Market_Cap': market_caps,
        'PE_Ratio': pe_ratios
    }
    #Robimy data frame z tego słownika i zapisyjemy w pliku .csv, jeżeli taki plik już jest to chyba powinien się nadpisać wsm nie sprawdzalem xd
    df = pd.DataFrame(data)
    df.to_csv("Technology-Sector-Data.csv")


def scrape_historical_data(header, symbol, save_dir):
    url = 'https://finance.yahoo.com/quote/'+ symbol +'/history?period1=1557842128&period2=1715694779'
    cookies = {
        'GUC': 'AQABCAFmQ8tmb0IhUATX&s=AQAAAJuCm0BM&g=ZkJ7Yg',
        'PRF': 't%3DGME',
        'GUCS': 'AXKF8mqr',
        'A3': 'd=AQABBFh7QmYCEFlEFuMkxA1dyXP0Op50YPUFEgABCAHLQ2ZvZu-bb2UBAiAAAAcIV3tCZm23rkw&S=AQAAAm7CQETmUWGD51w0N2mevQI',
        'A1': 'd=AQABBFh7QmYCEFlEFuMkxA1dyXP0Op50YPUFEgABCAHLQ2ZvZu-bb2UBAiAAAAcIV3tCZm23rkw&S=AQAAAm7CQETmUWGD51w0N2mevQI',
        'A1S': 'd=AQABBFh7QmYCEFlEFuMkxA1dyXP0Op50YPUFEgABCAHLQ2ZvZu-bb2UBAiAAAAcIV3tCZm23rkw&S=AQAAAm7CQETmUWGD51w0N2mevQI',
        'EuConsent': 'CP-jXAAP-jXAAAOACKENA0EgAAAAAAAAACiQAAAAAAAA',
    }

    initial_response = requests.get(url, headers=headers, allow_redirects=True, cookies=cookies)

    print(initial_response.is_redirect)
    print(initial_response.status_code)
    if 'consent' in initial_response.url:
        print('Consent required')
        return

    soup = BeautifulSoup(initial_response.text, 'html.parser')

    download_link = soup.find('a', attrs={"data-testid": "download-link"}).get('href')

    print(download_link)
    file_name = os.path.join(save_dir, symbol + '_historical_data.csv')
    with open(file_name, 'wb') as f:
        f.write(requests.get(download_link, headers=headers, cookies=cookies).content)

    print('File downloaded to: ', file_name)

def filter_data(files):
    for file in files:
        df = pd.read_csv('./data/' + file)
        df_reduced = df.iloc[750:]
        df_reduced.to_csv('./data/' + file, index=False)
