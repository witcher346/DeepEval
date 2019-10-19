import requests
import urllib.request
from bs4 import BeautifulSoup
import pandas as pd
from unidecode import unidecode

page=1
df = pd.DataFrame(columns=['img_url', 'price', 'page_url'])

try:

    while True:
        result=""
        url2="https://kiev.mesto.ua/sale/?currency=USD&p="+str(page)
        req = requests.get(url2)
        html_content2 = req.text
        soup = BeautifulSoup(html_content2,"html.parser")
        hrefs = soup.find_all('a',{'class' : 'title'})

        for href in hrefs:
            print(href.text)
            url=href['href']
            r = requests.get(url)
            html_content = r.text
            soup = BeautifulSoup(html_content,"html.parser")
            price = soup.find('span',{'class':"big"})
            links = soup.find_all('img',{'class' : 'img-responsive'})
            price = unidecode(price.get_text())
            price = price.replace(' ', '')
            pics = []
            for link in links:
                pics.append(link['src'])
            result = []
            for pic in pics:
                result.append([pic, price, url])
            mid_df = pd.DataFrame(result, columns=['img_url', 'price', 'page_url'])
            df = df.append(mid_df, ignore_index=True)
        df.to_excel('data_imgs.xlsx')
        page+=1
except Exception as e:
  print(e)










