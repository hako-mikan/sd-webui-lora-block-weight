# LoRA Block Weight
- custom script for [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 
- When applying Lora, strength can be set block by block.

- [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 用のスクリプトです
- Loraを適用する際、強さを階層ごとに設定できます

## 更新情報
2023.02.07 1250(JST)
- XYZプロットActive時の動作を変更しました(本体のScriptが優先されるようになります)

2023.02.06 2000(JST)
- 機能追加：XYZプロット機能を追加しました

2023.01.31 0200(JST)
- 機能追加：ランダム機能を追加しました
- 機能修正：ウェイトがマイナスにも効くようになりました

日本語説明は[後半](#概要)後半にあります。

## Updates
2023.02.07 1250(JST)
- Changed behavior when XYZ plot Active (Script of the main UI is prioritized).

2023.02.06 2000(JST)
- Feature added: XYZ plotting is added.

2023.01.31 0200(JST)
- Feature added: Random feature is added
- Fixed: Weighting now works for negative values.

# Summary
LoRA is a powerful tool, but it is sometimes difficult to handle and can affect even the areas you do not want it to affect. This script allows you to set the degree to which LoRA is applied at each level of the U-Net hierarchy. Using this script, you may be able to get the image you want.

## Usage
Place lora_block_weight.py in the script folder.  
Restart web-ui.bat on installation.　　
Put lbwpresets.txt in the same folder. It will work without it.

### Active  
Check here to enable.

### Prompt
On the prompt box, enter the LoRA you wish to use as usual. Next to the strength value, enter ":" and then the identifier. The identifier can be edited in the Weights setting.  
\<lora: "lora name":1:IN03>.  
The Lora strength is valid and applies to the entire hierarchy.

### Weights setting
Enter the identifier and weights.
Unlike the full model, Lora is divided into 17 blocks, including the encoder. Therefore, enter 17 values.
BASE, IN, OUT, etc. are the hierarchy equivalent to the full model.

|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|  
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|  
|BASE|IN01|IN02|IN04|IN05|IN07|IN08|MID|OUT03|OUT04|OUT05|OUT06|OUT07|OUT08|OUT09|OUT10|OUT11|

### Special values
Basically, a numerical value must be entered to work correctly, but by assuming "R" and "U", a random value is entered.  
R : Numerical value with 3 decimal places from 0~1
U : 3 decimal places from -1.5 to 1.5

For example, assuming ROUT:1,1,1,1,1,1,1,1,R,R,R,R,R,R,R,R,R  
Only the OUT layer is randomized  
The randomized values will be displayed on the command prompt screen when the image is generated.

The save button saves the text in the current text box. It is better to use a text editor, so use the open Texteditor button to open a text editor, edit the text, and reload it.  
The text box above the Weights setting is a list of currently available identifiers, useful for copying and pasting into an XY plot. 17 values are required to appear in the list.

### Fun Usage
Assuming you are using this in conjunction with the XY plot(Built-in features of Automatic1111.), you can examine the impact of each level of the hierarchy.  
![xy_grid-0017-4285963917](https://user-images.githubusercontent.com/122196982/215341315-493ce5f9-1d6e-4990-a38c-6937e78c6b46.jpg)

The setting values are as follows.  
NOT:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0  
ALL:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1  
INS:1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0  
IND:1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0  
INALL:1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0  
MIDD:1,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0  
OUTD:1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0  
OUTS:1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1  
OUTALL:1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1 

## XYZ Plotting Function
The optimal value can be searched for on a round-robin basis by depending on the value of each layer individually.
### Usage 
This function works by checking the Active checkbox. If a script (such as XYZ plot in Automatic1111) is enabled, it will take precedence.　　 Hires. fix is not supported. batch size is fixed to 1. batch count should be set to 1.  
Enter XYZ as the identifier of the LoRA that you want to depend on. It will work even if you do not enter a value corresponding to XYZ in the preset. If a value corresponding to XYZ is entered, that value will be used as the initial value.
### Axis type
#### value
Sets the weight of the hierarchy to depend on. Enter the values separated by commas. 0,0.25,0.5,0.75,1", etc.

#### BLock ID
Assuming a block ID, only that block will change to the value specified by value. As with the other types, use commas to separate them. Multiple blocks can also depend simultaneously by separating them with a space or hyphen. The first NOT character inverts the change: NOT IN08-OUT05 will depend on all blocks except IN08-OUT05.

#### Seed
Seed changes, and is intended to be specified on the Z-axis.

#### Original Weight
Specify the initial value that depends on the weight of each block. If Original Weight is enabled, the value entered for XYZ is ignored.

### Input example
X : value, value : 0,0.25,0.5,0.75,1  
Y : Block ID, value : BASE,IN01-IN08,IN05-OUT05,OUT03-OUT11,NOT OUT03-OUT11  
Z : Original Weights, Value : NONE,ALL0.5,ALL  

In this case, an XY plot is created corresponding to the initial values NONE,ALL0.5,ALL.
If you select Seed for Z and enter -1,-1,-1, three XY plots will be created with different seeds.


For more information on hierarchical merging, please refer to

https://github.com/bbc-mc/sdweb-merge-block-weighted-gui

# 概要
Loraは強力なツールですが、時に扱いが難しく、影響してほしくないところにまで影響がでたりします。このスクリプトではLoraを適用する際、適用度合いをU-Netの階層ごとに設定することができます。これを使用することで求める画像に近づけることができるかもしれません。

## 使い方
scriptフォルダにlora_block_weightを置いてください。  インストール時はWeb-ui.batを再起動をしてください。
lbwpresets.txtも同じフォルダに入れてください。なくても動きます。

### Active  
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

### 特別な値
基本的には数値を入れないと正しく動きませんが R および U を入力することでランダムな数値が入力されます。  
R : 0~1までの小数点3桁の数値
U : -1.5～1.5までの小数点3桁の数値

例えば　ROUT:1,1,1,1,1,1,1,1,R,R,R,R,R,R,R,R,R  とすると  
OUT層のみダンダム化されます  
ランダム化された数値は画像生成時にコマンドプロンプト画面に表示されます

saveボタンで現在のテキストボックスのテキストを保存できます。テキストエディタを使った方がいいので、open Texteditorボタンでテキストエディタ開き、編集後reloadしてください。  
Weights settingの上にあるテキストボックスは現在使用できる識別子の一覧です。XYプロットにコピペするのに便利です。17個ないと一覧に表示されません。

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

## XYZ プロット機能
各層の値を個別に変化させることで最適値を総当たりに探せます。
### 使い方 
Activeをチェックすることで動作します。 Script(Automatic1111本体のXYZプロットなど)が有効になっている場合そちらが優先されます。noneを選択してください。
Hires. fixには対応していません。Batch sizeは1に固定されます。Batch countは1に設定してください。  
変化させたいLoRAの識別子にXYZと入力します\<lora:"lora名":1:XYZ>。 プリセットにXYZに対応する値を入力していなくても動作します。その場合すべてのウェイトが0の状態からスタートします。XYZに対応する値が入力されている場合はその値が初期値になります。
### 軸タイプ
#### value
変化させる階層のウェイトを設定します。カンマ区切りで入力してください。「0,0.25,0.5,0.75,1」など。

#### BLock ID
ブロックIDを入力すると、そのブロックのみvalueで指定した値に変わります。他のタイプと同様にカンマで区切ります。スペースまたはハイフンで区切ることで複数のブロックを同時に変化させることもできます。最初にNOTをつけることで変化対象が反転します。NOT IN09-OUT02とすると、IN09-OUT02以外が変化します。NOTは最初に入力しないと効果がありません。IN08-M00-OUT03は繋がっています。

#### Seed
シードが変わります。Z軸に指定することを想定しています。

#### Original Weight
各ブロックのウェイトを変化させる初期値を指定します。プリセットに登録されている識別子を入力してください。Original Weightが有効になっている場合XYZに入力された値は無視されます。

### 入力例
X : value, 値 : 1,0.25,0.5,0.75,1  
Y : Block ID, 値 : BASE,IN01-IN08,IN05-OUT05,OUT03-OUT11,NOT OUT03-OUT11  
Z : Original Weights, 値 : NONE,ALL0.5,ALL  

この場合、初期値NONE,ALL0.5,ALLに対応したXY plotが作製されます。
ZにSeedを選び、-1,-1,-1を入力すると、異なるseedでXY plotを3回作製します。


階層別マージについては下記を参照してください

https://github.com/bbc-mc/sdweb-merge-block-weighted-gui
