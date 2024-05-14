import unittest
import os
from scraper import scrape_historical_data, header
import pandas as pd

class Tests(unittest.TestCase):
    
    def test_historical_data_scraping(self):
        df = pd.read_csv('Technology-Sector-Data.csv')
        symbols = df['Symbol']

        for symbol in symbols:
            scrape_historical_data(header, symbol, './test-data')

        #Sprawdzenie czy pobrany funkcja scrape_historical_data została wykonana i zapisała plik z danymi dla każdego symbolu
        files_downloaded = os.listdir('./test-data')
        self.assertEqual(len(files_downloaded), len(symbols))

        #Sprawdzenie czy każdy plik ma w sobie jakąś zawartość
        for file_name in files_downloaded:
            file_path = os.path.join('./test-data', file_name)
            self.assertTrue(os.path.getsize(file_path) > 0, f"Plik {file_name} jest pusty.")


if __name__ == "__main__":
    unittest.main()