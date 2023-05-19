import os
from PIL import Image

#ファイル名を宣言
FromImgName = 'orig'
ToImgName = 'resize'

#imgフォルダ内の画像名をまとめて取得
files = os.listdir(FromImgName)

#for文で画像サイズを一括変更
for file in files:
    img = Image.open(os.path.join(FromImgName, file)) #画像のパスを生成し、imgへ画像を格納
    img_resize = img.resize((283, 215), Image.LANCZOS)  #横75mm 96dpi
    filename = 'resize_' + file  # 保存するファイル名に'resize_'を追加
    img_resize.save(os.path.join(ToImgName, filename), quality = 100)  #resizeフォルダへ保存
