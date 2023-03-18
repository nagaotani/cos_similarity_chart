import streamlit as st
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import mplfinance as mpf
import math
import datetime
from pandas_datareader import data

st.title('類似チャートパターン検索')
st.header('日経平均の過去５日分のチャートを作成し過去の類似度の高いチャートを検索する')

# pandas_datareaderでStooqから日経平均の価格を取得

start_time = st.slider('何年分のデータを利用しますか？', 5, 50, 33)

end = datetime.datetime.today()
start = end -datetime.timedelta(days=start_time*365.25)
df = data.DataReader('^NKX', 'stooq', start, end)

st.write('データのスタート日：', start.date())

# 計算しやすい（自分にとって）ように日付の並びを変更し、入力しやすいように文字列を小文字にする
df = df.sort_index(ascending=False)
df.index.name = 'date'
df.columns = ['open', 'high', 'low', 'close', 'volume']

# 必要なデータ（４本値）でＤＦを作成

df = df.iloc[:,:4]

# window日分の４本値の組み合わせをつくる

window = 5 # 何日分のローソク足を取ってくるか

x_data = [] # x_dataにwindow日分の４本値のarrayを格納する

for i in range(len(df)-window):  
  x = df.iloc[i:window+i, :].values # i番目からi+5番目までの４本値
  x = x.reshape(window * 4) # １次元にする
  x = preprocessing.minmax_scale(x) # 最大値を１、最小値を０にして正規化
  x = x.reshape(window, 4) # 正規化後、もう一度４本値Xwindow日のデータサイズに戻す
  x_data.append(x) # X_dataに格納する

# コサイン類似度
def cos_sim(X,Y):
    return np.dot(X,Y)/(np.linalg.norm(X)*np.linalg.norm(Y))

# コサイン類似度を求める
i = 0
cossim_result=[] # 結果を格納するリスト

X = x_data[0].reshape(window*4) # 検索したい元データの作成

for y in x_data:
  date = df.index[i]
  Y = y.reshape(window*4)
  cossim_result.append([date, cos_sim(X, Y)])  # 日付とコサイン類似度を格納
  i = i + 1

# コサイン類似度をソート
cossim_result_sorted = sorted(cossim_result, key=lambda x: x[1], reverse=True)


# 直近のデータ分のグラフの作成
st.header(f'直近のデータ日付：{df.iloc[0].name.strftime("%Y年%m月%d日")}')
df_today = df[:window].sort_index()
fig = mpf.figure(figsize=(5, 5),style='checkers')
ax1 = fig.add_subplot(1, 1, 1)
mpf.plot(df_today, ax=ax1, style='checkers', type='candle', xrotation=30, datetime_format='%Y/%m/%d')
st.pyplot(fig)

# 基準日からwindow日の過去データと、基準日からfuture_days日の未来データの４本値を作る
dfs_origin = [] # 基準日からwindow日分遡るデータの格納リスト
dfs_future = [] # 基準日からwindow日分遡るデータと基準日からfutre_days分の未来データの格納リスト
future_days = 5 # 何日分の未来データを取るか

graph_number = 5 # 上位何個分の類似チャートを表示するか

for i in range(1, graph_number+1):
  n = df.index.get_loc(cossim_result_sorted[i][0])
  d = df.iloc[n:(n + window), :].sort_index()

  dfs_origin.append(d)
  if n >= future_days:
    d = df.iloc[(n - future_days):(n + window), :].sort_index()
  else:
    d = df.iloc[:(n + window),:].sort_index()
  dfs_future.append(d)


i = 1
for dfo, dff in zip(dfs_origin, dfs_future):
  fig = mpf.figure(figsize=(5, 5),style='checkers')
  ax1 = fig.add_subplot(2, 2, 1)
  ax2 = fig.add_subplot(2, 2, 2)
  plt.subplots_adjust(wspace=1.0, hspace=0.6)
  mpf.plot(dfo, ax=ax1, style='checkers', type='candle', xrotation=30, datetime_format='%Y/%m/%d')
  mpf.plot(dff, ax=ax2, style='checkers', type='candle', xrotation=30, datetime_format='%Y/%m/%d')
  st.subheader(f'過去の類似日{i}：{dff.iloc[window-1].name.strftime("%Y年%m月%d日")}')
  st.pyplot(fig)

  st.write('\n')
  st.write(f'翌日寄付で買い、翌日引けで売り: {((dff.iloc[window].close - dff.iloc[window].open)/dff.iloc[window].open*100):+.2f}%')
  st.write(f'翌日寄付で買い、１日後の引けで売り: {(dff.iloc[window+1].close - dff.iloc[window].open)/dff.iloc[window].open*100:.2f}%')
  st.write(f'翌日寄付で買い、２日後の引けで売り: {(dff.iloc[window+2].close - dff.iloc[window].open)/dff.iloc[window].open*100:.2f}%')
  st.write(f'翌日寄付で買い、３日後の引けで売り: {(dff.iloc[window+3].close - dff.iloc[window].open)/dff.iloc[window].open*100:.2f}%')
  st.write(f'翌日寄付で買い、４日後の引けで売り: {(dff.iloc[window+4].close - dff.iloc[window].open)/dff.iloc[window].open*100:.2f}%')
#  st.write('\n')
  i += 1

# 類似上位ｎ個の平均リターンテーブル作成

top_n = 20 # 上位何個取ってくるか
future_days = 5 # 何日先までリターンを調べるか

# 変数の初期設定
day1_return = 0
day2_return = 0
day3_return = 0
day4_return = 0
day5_return = 0

day1_win = 0
day2_win = 0
day3_win = 0
day4_win = 0
day5_win = 0

data = []
win = []

for i in range(1, top_n + 1):
  n = df.index.get_loc(cossim_result_sorted[i][0])
  if n >= future_days:
    d = df.iloc[(n - future_days):(n + window), :].sort_index()

  day1 = (d.iloc[window].close - d.iloc[window].open)/d.iloc[window].open
  day2 = (d.iloc[window+1].close - d.iloc[window].open)/d.iloc[window].open
  day3 = (d.iloc[window+2].close - d.iloc[window].open)/d.iloc[window].open
  day4 = (d.iloc[window+3].close - d.iloc[window].open)/d.iloc[window].open
  day5 = (d.iloc[window+4].close - d.iloc[window].open)/d.iloc[window].open

  day1_return += day1
  day2_return += day2
  day3_return += day3
  day4_return += day4
  day5_return += day5

  if day1 > 0:
    day1_win += 1
  if day2 > 0:
    day2_win += 1
  if day3 > 0:
    day3_win += 1
  if day4 > 0:
    day4_win += 1
  if day5 > 0:
    day5_win += 1

  if i % 5 == 0:
    # st.subheader(f'類似度上位{i}日分の平均値')
    # st.write(f'１日目のリターン：{day1_return*100/i:.2f}%')
    # st.write(f'２日目のリターン：{day2_return*100/i:.2f}%')
    # st.write(f'３日目のリターン：{day3_return*100/i:.2f}%')
    # st.write(f'４日目のリターン：{day4_return*100/i:.2f}%')
    # st.write(f'５日目のリターン：{day5_return*100/i:.2f}%')

    df_data =[]
    df_data.append(f'{day1_return*100/i:.3f}')
    df_data.append(f'{day2_return*100/i:.3f}')
    df_data.append(f'{day3_return*100/i:.3f}')
    df_data.append(f'{day4_return*100/i:.3f}')
    df_data.append(f'{day5_return*100/i:.3f}')
    data.append(df_data)

    df_win = []
    df_win.append(day1_win)
    df_win.append(day2_win)
    df_win.append(day3_win)
    df_win.append(day4_win)
    df_win.append(day5_win)
    win.append(df_win)

st.write('------------------------------------------------------------')

df_result = pd.DataFrame(data=data,
                         index = ['コサイン類似度の上位５パターン分','コサイン類似度の上位１０パターン分','コサイン類似度の上位１５パターン分','コサイン類似度の上位２０パターン分'] ,
                         columns=['１日目の平均リターン', '２日目の平均リターン', '３日目の平均リターン', '４日目の平均リターン', '５日目の平均リターン',])
st.subheader('平均リターンの計算（パーセント）')
st.table(df_result)
st.write('\nデイトレは当日朝の寄付で買い、当日引けで売り\nその他は当日朝の寄付で買い、N日後の引けで売り')

df_win_rate = pd.DataFrame(data=win,
                         index = ['コサイン類似度の上位５日分','コサイン類似度の上位１０日分','コサイン類似度の上位１５日分','コサイン類似度の上位２０日分'] ,
                         columns=['１日目の勝ち数', '２日目の勝ち数', '３日目の勝ち数', '４日目の勝ち数', '５日目の勝ち数',])
st.subheader('勝ち数')
st.table(df_win_rate)
st.write('\nデイトレは当日朝の寄付で買い、当日引けで売り\nその他は当日朝の寄付で買い、N日後の引けで売り')