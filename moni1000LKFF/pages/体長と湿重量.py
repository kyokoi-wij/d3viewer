import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px

from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



# 0. ページタイトルと表示設定
image4 = Image.open('./moni1000LKFF/fig/site_logo.png') 
st.set_page_config(page_title = 'Moni1000 LKFF SL-WW app', page_icon = image4, layout = 'wide')
st.title('体長と湿重量の相関')



try:
    # Layout (Sidebar)↓
    # データを読み込む   
    st.sidebar.subheader('🤖データのアップロード')
    uploaded_file5 = st.sidebar.file_uploader('KOS06-2 の csvファイルをアップロードしてください。', 
                                              type = 'csv', 
                                              key = 'up_slww')
    # Layout (Sidebar)↑
    

    # データフレームの作成
    if uploaded_file5 is not None:

        df_ffsw = pd.read_csv(uploaded_file5, comment = "#")
        df_ffsw['site_code'] = df_ffsw['catalog_number'].str[7:10]
        df_slww = df_ffsw[['site_code',
                           'year_collected',
                           'japanese_name',
                           'standard_length',
                           'wet_weight',
                           'sex']]
        df_slww.replace({'ND': np.nan, 'N/A': np.nan, 'nd': np.nan, 'na': np.nan}, inplace=True)
        df_slww['year_collected'] = 'FY ' + df_slww['year_collected'].astype(str)
        df_slww['standard_length'] = df_slww['standard_length'].astype('float64')
        df_slww['wet_weight'] = df_slww['wet_weight'].astype('float64')
        df_slww['log_standard_length'] = np.log(df_slww['standard_length'])
        df_slww['log_wet_weight'] = np.log(df_slww['wet_weight'])
       
    st.write('#')



    # Layout (Sidebar)↓
    sp_list = list(df_slww['japanese_name'].unique())
    selected_sp = st.sidebar.selectbox('表示する種を選択', sp_list)
    # 選択された種名でデータをフィルタリング
    df_slww_sp = df_slww[(df_slww['japanese_name'] == selected_sp)]
    
    site_list = list(df_slww['site_code'].unique())
    selected_site = st.sidebar.multiselect('どのサイトのデータを用いるか選択', site_list, default = site_list) 
    # 選択されたサイトでデータをフィルタリング
    df_slww_sp_site = df_slww_sp[(df_slww_sp['site_code'].isin(selected_site))]
    # Layout (Sidebar)↑



    # 体長と湿重量の元データを用いてアロメトリーのグラフを描画
    st.write('#')
    st.subheader('1. 体長と湿重量の散布図と回帰曲線')
    scatter = alt.Chart(df_slww_sp_site).mark_point(
        filled=True, size=150
        ).encode(
        x = alt.X('standard_length', 
                axis = alt.Axis(labelFontSize=12, ticks=True, titleFontSize=14, title='体長（mm）',labelAngle=0)),
        y = alt.Y('wet_weight', 
                axis = alt.Axis(labelFontSize=12, ticks=True, titleFontSize=14, title='湿重量（g）',labelAngle=0)),
        color = alt.Color('site_code', 
                        legend = alt.Legend(title ='サイトコード', orient = 'top-left')))
        
    regress = scatter + scatter.transform_regression('standard_length', 'wet_weight', method='exp').mark_line(
        shape='mark'
        ).encode(
        color = alt.Color('site_code'))


    st.altair_chart(regress, theme = None, use_container_width = True)
    


    # 体長と湿重量のデータを対数変換して直線回帰のグラフを表示
    st.write('#')
    st.subheader('2. log体長とlog湿重量の散布図と回帰直線')
    scatter_trace = px.scatter(df_slww_sp_site, 
                            x='log_standard_length', 
                            y='log_wet_weight', 
                            trendline=None,
                            color='site_code')
    
    scatter_trace.update_layout(
        xaxis_title="ln 体長",
        yaxis_title="ln 湿重量",
        font=dict(size=20),
        legend=dict(title="サイトコード", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(scatter_trace, use_container_width = True)



    # 回帰式のパラメーターを利用して体長から湿重量を推定するウィジェットを作成
    # モデルのインスタンスを作成し、説明変数と目的変数を指定
    X = df_slww_sp_site[['log_standard_length']]
    y = df_slww_sp_site['log_wet_weight']
    model = LinearRegression()
    model.fit(X, y)

    # 回帰直線のパラメータを取得
    slope = model.coef_[0]  # 傾き
    intercept = model.intercept_  # 切片
    
    # 回帰式を表示
    st.write('#')
    st.subheader('3. 体長から湿重量を推定')
    st.write('#')
    st.write(f'直線の回帰式： ln (y) = {slope:.2f} ln (x) + {intercept:.2f}')
    
    # 決定係数を表示
    y_pred = model.predict(X)  # 予測値
    r_squared = r2_score(y, y_pred)
    st.write(f'決定係数（R2）： {r_squared:.3f}')
    st.write('#')
    
    # 'standard_length' の値を入力
    new_X = st.number_input('体長（mm）を入力', value = 100.0, step = 1.0)
    
    # 'standard_length' の値に対応する 'wet_weight' を推定
    predicted_y = round(float(np.exp(model.predict([[np.log(new_X)]]))), 2)
    
    # 推定された 'wet_weight' の値を表示
    st.subheader(f'推定湿重量： {predicted_y} g')
    


    # Layout (Sidebar)↓
    st.sidebar.write('#')
    st.sidebar.write('# #基本情報#')
    st.sidebar.write(f'### サンプルサイズ：  {df_slww_sp_site.shape[0]}')
    st.sidebar.write(f'### カラム数     ：  {df_slww_sp_site.shape[1]}')
    st.sidebar.write('#### 各カラムの欠損値数：')

    # 欠損値の表示
    null_df = pd.DataFrame(df_slww_sp_site.isnull().sum(), columns=['欠損値数'])
    st.sidebar.dataframe(null_df)
    # Layout (Sidebar)↑



except:
    st.sidebar.error('データを読み込んでください。データを読み込めない場合は、\
                     数値データに文字列データが混じっている可能性があります。', icon="🚨")















    # NumPyを用いて回帰直線の係数を求め体長から湿重量を推定 
    # X = df_slww_sp_site['standard_length']
    # Y = df_slww_sp_site['wet_weight']
    # coef_plt = np.polyfit(X, np.log(Y), 1)
    # df_slww_sp_site['log_wet_weight_pred'] = coef_plt[1] + coef_plt[0] * X
    # df_slww_sp_site['wet_weight_pred'] = np.exp(df_slww_sp_site['log_wet_weight_pred'])

    # 決定係数を算出
    # ss_res = np.sum((np.log(Y) - df_slww_sp_site['log_wet_weight_pred']) ** 2)
    # ss_tot = np.sum((np.log(Y) - np.mean(np.log(Y))) ** 2)
    # r_squared = 1 - (ss_res / ss_tot)

    # 回帰式の表示
    # st.text(f'y = {np.exp(coef_plt[1]):.5f} * e^({coef_plt[0]:.5f}x)')
    # 結果を表示
    # result_df = pd.DataFrame({'決定係数': [r_squared]})
    # st.write('決定係数： {:.3f}'.format(result_df.iloc[0]['決定係数']))


    # 回帰式を用いて体長から湿重量を求める
    # input_length = st.number_input('体長（mm）を入力してください。', min_value=df_slww_sp_site.standard_length.min(), max_value=df_slww_sp_site.standard_length.max(), value=100.0, step=0.1)
    
    # Calculate Wet Weight using Regression Equation
    # coef_plt = np.polyfit(X, np.log(Y), 1) # Use the same coefficients from previous code block
    # log_wet_weight_pred = coef_plt[1] + coef_plt[0] * input_length
    # wet_weight_pred = np.exp(log_wet_weight_pred)

    # Display Predicted Wet Weight
    # st.write(f'予測湿重量: {wet_weight_pred:.5f} g')
