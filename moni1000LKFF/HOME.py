import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import folium    



#streamlit run app.py
image4 = Image.open('./moni1000LKFF/fig/site_logo.png') 
st.set_page_config(page_title = 'Moni1000 LK_FF app', page_icon = image4, layout = 'wide')

col1, col2 = st.columns(2)

with col1:
        image1 = Image.open('./moni1000LKFF/fig/WIlogo.png') 
        st.image(image1, width=200)

with col2:
        image2 = Image.open('./moni1000LKFF/fig/monilogo.png') 
        st.image(image2, width=200)



st.title('モニ1000陸水域：湖沼 淡水魚類調査')
st.subheader('Biodiversity Dashboard by WIJ')
st.write("#")
st.write('このページは、モニタリングサイト1000 陸水域 湖沼 淡水魚類調査のデータをインタラクティブに利用するため、\
    PythonのStreamlitライブラリを使用して開発したウェブアプリケーションです。\
    日本国際湿地保全連合　2023年6月')
st.write("#")



# 地図の表示。
site_gps = {
    'res': ['調査速報を見る', '調査速報を見る', '調査速報を見る', '調査速報を見る', '調査速報を見る', '調査速報を見る', '調査速報を見る', '調査速報を見る', '調査速報を見る', '調査速報を見る', '調査速報を見る'],
    'site': ['ウトナイ湖', '達古武湖', '屏風山湖沼群', '伊豆沼・内沼', '猪苗代湖', '西浦古渡', '北浦爪木', '三方湖', '琵琶湖', '宍道湖', '鎮西湖'],
    'lat': [42.7, 43.1, 40.81, 38.71, 37.48, 35.98, 35.97, 35.56, 35.44, 35.45, 33.33],
    'lon': [141.69, 144.48, 140.27, 141.1, 140.1, 140.36, 140.6, 135.89, 136.18, 132.94, 130.6],
    'url':['https://www.biodic.go.jp/moni1000/findings/newsflash/pdf/lakes_freshwaterfishes_h30.pdf',
           'https://www.biodic.go.jp/moni1000/findings/newsflash/pdf/lakes_freshwaterfishes_h30.pdf',
           'https://www.biodic.go.jp/moni1000/findings/newsflash/pdf/freshwaterfishes_2019.pdf',
           'https://www.biodic.go.jp/moni1000/findings/newsflash/pdf/freshwaterfishes_2020.pdf',
           'https://www.biodic.go.jp/moni1000/findings/newsflash/pdf/freshwaterfishes_2019.pdf',
           'https://www.biodic.go.jp/moni1000/findings/newsflash/pdf/freshwaterfishes_2020.pdf',
           'https://www.biodic.go.jp/moni1000/findings/newsflash/pdf/freshwaterfishes_2020.pdf',
           'https://www.biodic.go.jp/moni1000/findings/newsflash/pdf/lakes_freshwaterfishes_h29.pdf',
           'https://www.biodic.go.jp/moni1000/findings/newsflash/pdf/freshwaterfishes_2022.pdf',
           'https://www.biodic.go.jp/moni1000/findings/newsflash/pdf/lakes_freshwaterfishes_h29.pdf',
           'https://www.biodic.go.jp/moni1000/findings/newsflash/pdf/freshwaterfishes_2022.pdf']
}

df_gps = pd.DataFrame(site_gps)


# 地図の中心の緯度/経度、タイル、初期のズームサイズ。
m = folium.Map(
    # 地図の中心位置の指定(
    location=[38.7163, 141.1044], 
    # タイル、アトリビュートの指定
    tiles='https://cyberjapandata.gsi.go.jp/xyz/pale/{z}/{x}/{y}.png',
    attr='モニ1000陸水域 湖沼調査',
    # ズームを指定
    zoom_start = 4
)



# 読み込んだデータ(緯度・経度、ポップアップ用文字、アイコンを表示)
for i, row in df_gps.iterrows():
    # ポップアップの作成
    pop=row['url']
    folium.Marker(
        # 緯度と経度を指定
        location=[row['lat'], row['lon']],
        # ツールチップの指定(都道府県名)
        tooltip=f"{row['site']}サイト",
        # ポップアップの指定
        popup=folium.Popup(f"<a href='{row['url']}' target='_blank'>{row['res']}</a>", max_width=300),
        # アイコンの指定(アイコン、色)
        icon=folium.Icon(icon="eye-open",icon_color="white", color="blue")
    ).add_to(m)

st_data = st_folium(m, width=700, height=500)




st.write('https://www.biodic.go.jp/moni1000/site_list.html')
