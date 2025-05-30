# Import library yang dibutuhkan
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Buatlah request ke website
website_url = requests.get('https://id.wikipedia.org/wiki/Demografi_Indonesia').text
soup = BeautifulSoup(website_url, 'html.parser')

# Ambil table dengan class 'wikitable sortable'
my_table = soup.find('table', {'class': 'wikitable sortable'})

# Cari data dengan tag 'td'
links = my_table.findAll('td')

# Buatlah lists kosong 
nama = []
luas_km = []
populasi10 = []
populasi20 = []

# Memasukkan data ke dalam list berdasarkan pola HTML
for i, link in enumerate(links):
    if i in range(0, len(links), 4):
        nama.append(link.get_text()[:-1])
    if i in range(1, len(links), 4):
        luas_km.append(link.get_text()[:-1])
    if i in range(2, len(links), 4):
        populasi10.append(link.get_text()[:-1])
    if i in range(3, len(links), 4):
        populasi20.append(link.get_text()[:-1])

# Buatlah DataFrame dan masukkan ke CSV
df = pd.DataFrame()
df['Nama Provinsi'] = nama
df['Luas km'] = luas_km
df['Populasi 2010'] = populasi10
df['Populasi 2020'] = populasi20
df.to_csv('Indonesia_Demography_by_Province.csv', index=False, encoding='utf-8', quoting=1)

# Tampilkan DataFrame
print(df)

# -----------------------------

# Import library yang dibutuhkan
import re

# Function email_check
def email_check(input):
    match = re.search(r'(?=^((?!-).)*$)(?=[^0-9])((?=^((?!\.\d).)*$)|(?=.*_))', input)
    if match:
        print('Pass')
    else:
        print('Not Pass')

# Masukkan data email ke dalam list
emails = [
    'my-name@someemail.com', 
    'myname@someemail.com', 
    'my.name@someemail.com', 
    'my.name2019@someemail.com', 
    'my.name.2019@someemail.com', 
    'somename.201903@someemail.com', 
    'my_name.201903@someemail.com', 
    '201903myname@someemail.com', 
    '201903.myname@someemail.com'
]

# Looping untuk pengecekan Pass atau Not Pass
for email in emails:
    email_check(email)
