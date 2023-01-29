# Lora Block Weight
- custom script for [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 
- When applying Lora, strength can be set block by block.

- [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 用のスクリプトです
- Loraを適用する際、強さを階層ごとに設定できます

# 概要
Loraは強力なツールですが、時に扱いが難しく、影響してほしくないところにまで影響がでたりします。このスクリプトではLoraを適用する際、適用度合いをU-Netの階層ごとに設定することができます。これを使用することで求める画像に近づけることができるかもしれません。

## 使い方
scriptフォルダにlora_bw.pyを置いてください。  
lbwpresets.txtも同じフォルダに入れてください。なくても動きます。

### use Block  
ここにチェックを入れることで動作します。

### プロンプト
プロンプト画面では通常通り使用したいLoraを記入してください。その際、強さの値の次に「:」を入力し次に識別子を入力します。識別子はWeights setting で編集します。  
\<lora:"lora名":1:IN03>
Loraの強さは有効で、階層全体にかかります。

### Weights setting
識別子とウェイトを入力します。
フルモデルと異なり、Loraではエンコーダーを含め17のブロックに分かれています。よって、17個の数値を入力してください。
BASE,IN,OUTなどはフルモデル相当の階層です。

|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|  
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|  
|BASE|IN01|IN02|IN04|IN05|IN07|IN08|MID|OUT03|OUT04|OUT05|OUT06|OUT07|OUT08|OUT09|OUT10|OUT11|

saveボタンで現在のテキストボックスのテキストを保存できます。テキストエディタを使った方がいいので、open Texteditorボタンでテキストエディタ開き、編集後reloadしてください。  
Weights settingの上にあるテキストボックスは現在使用できる識別子の一覧です。XYプロットにコピペするのに便利です。一覧にはしていますが、使えるかどうかはチェックしていません。17個ないと動きません。


### 楽しい使い方
XY plotと併用することで各階層の影響を調べることが可能になります。  
![xy_grid-0017-4285963917](https://user-images.githubusercontent.com/122196982/215341315-493ce5f9-1d6e-4990-a38c-6937e78c6b46.jpg)

設定値は以下の通りです。  
NOT:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0  
ALL:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1  
INS:1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0  
IND:1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0  
INALL:1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0  
MIDD:1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0  
OUTD:1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0  
OUTS:1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1  
OUTALL:1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1  

階層別マージについては下記を参照してください

https://github.com/bbc-mc/sdweb-merge-block-weighted-gui
