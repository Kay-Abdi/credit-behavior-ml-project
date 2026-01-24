import requests
import pandas as pd


import requests
import pandas as pd

url = "https://api.worldbank.org/v2/country/USA/indicator/FS.AST.PRVT.GD.ZS?format=json&per_page=60"

response = requests.get(url)
data = response.json()

records = []
for item in data[1]:
    records.append({
        "year": item["date"],
        "credit_private_sector_pct_gdp": item["value"]
    })

df_world_bank = pd.DataFrame(records)
df_world_bank = df_world_bank.dropna()

