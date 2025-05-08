import requests

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=CRYPTO:BTC&time_from=20250410T0130&limit=10000&sort=LATEST&apikey=683SUL8BPL2RK67T'
r = requests.get(url)
data = r.json()

print(data)