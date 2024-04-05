# LoRA Block Weight
- custom script for [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 
- When applying Lora, strength can be set block by block.

- [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 用のスクリプトです
- Loraを適用する際、強さを階層ごとに設定できます

[<img src="https://img.shields.io/badge/lang-Egnlish-red.svg?style=plastic" height="25" />](#overview)
[<img src="https://img.shields.io/badge/言語-日本語-green.svg?style=plastic" height="25" />](#概要)
[<img src="https://img.shields.io/badge/Support-%E2%99%A5-magenta.svg?logo=github&style=plastic" height="25" />](https://github.com/sponsors/hako-mikan)

> [!IMPORTANT]
> If you have an error :`ValueError: could not convert string to float`  
> use new syntax`<lora:"lora name":1:lbw=IN02>`


## Updates/更新情報
### 2024.04.06.0000(JST)
- add [new UI](#make-weights): make weights
- ウェイトを作成する[新しいUI](#ウェイトの作成)を追加

### 2023.11.22.2000(JST)
- bugfix
- added new feature:start in steps
- 機能追加:LoRAの途中開始

### 2023.11.21.1930(JST)
- added new feature:stop in steps
- 機能追加:LoRAの途中停止  
By specifying `<lora:"lora name":lbw=ALL:stop=10>`, you can disable the effect of LoRA at the specified step. In the case of character or composition LoRA, a sufficient effect is achieved in about 10 steps, and by cutting it off at this point, it is possible to minimize the impact on the style of the painting  
`<lora:"lora name":lbw=ALL:stop=10>`と指定することで指定したstepでLoRAの効果を無くします。キャラクターや構図LoRAの場合には10 step程度で十分な効果があり、ここで切ることで画風への影響を抑えることが可能です。

# Overview
Lora is a powerful tool, but it is sometimes difficult to use and can affect areas that you do not want it to affect. This script allows you to set the weights block-by-block. Using this script, you may be able to get the image you want.

## Usage
Place lora_block_weight.py in the script folder.  
Or you can install from Extentions tab in web-ui. When installing, please restart web-ui.bat.

### Active  
Check this box to activate it.

### Prompt
In the prompt box, enter the Lora you wish to use as usual. Enter the weight or identifier by typing ":" after the strength value. The identifier can be edited in the Weights setting.  
```
<lora:"lora name":1:0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0>.  
<lora:"lora name":1:0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0>.  (a1111-sd-webui-locon, etc.)
<lyco:"lora name":1:1:lbw=IN02>  (a1111-sd-webui-lycoris, web-ui 1.5 or later)
<lyco:"lora name":1:1:lbw=1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0>  (a1111-sd-webui-lycoris, web-ui 1.5 or later)
```
For LyCORIS using a1111-sd-webui-lycoris, syntax is different.
`lbw=IN02` is used and follow lycoirs syntax for others such as unet or else.
a1111-sd-webui-lycoris is under under development, so this syntax might be changed. 

Lora strength is in effect and applies to the entire Blocks.  
It is case-sensitive.
For LyCORIS, full-model blobks used,so you need to input 26 weights.
You can use weight for LoRA, in this case, the weight of blocks not in LoRA is set to 0.　　
If the above format is not used, the preset will treat it as a comment line.

### start, stop step
By specifying `<lora:"lora name":lbw=ALL:start=10>`, the effect of LoRA appears from the designated step. By specifying `<lora:"lora name":lbw=ALL:stop=10>`, the effect of LoRA is eliminated at the specified step. In the case of character or composition LoRA, a significant effect is achieved in about 10 steps, and by cutting it off at this point, it is possible to minimize the influence on the style of the painting. By specifying `<lora:"lora name":lbw=ALL:step=5-10>`, LoRA is activated only between steps 5-10."

### Weights Setting
Enter the identifier and weights.
Unlike the full model, Lora is divided into 17 blocks, including the encoder. Therefore, enter 17 values.
BASE, IN, OUT, etc. are the blocks equivalent to the full model.
Due to various formats such as Full Model and LyCORIS and SDXL, script currently accept weights for 12, 17, 20, and 26. Generally, even if weights in incompatible formats are inputted, the system will still function. However, any layers not provided will be treated as having a weight of 0.

LoRA(17)
|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|  
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|  
|BASE|IN01|IN02|IN04|IN05|IN07|IN08|MID|OUT03|OUT04|OUT05|OUT06|OUT07|OUT08|OUT09|OUT10|OUT11|

LyCORIS, etc.  (26)
|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|  
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|BASE|IN00|IN01|IN02|IN03|IN04|IN05|IN06|IN07|IN08|IN09|IN10|IN11|MID|OUT00|OUT01|OUT02|OUT03|OUT04|OUT05|OUT06|OUT07|OUT08|OUT09|OUT10|OUT11|

SDXL LoRA(12)
|1|2|3|4|5|6|7|8|9|10|11|12|
|-|-|-|-|-|-|-|-|-|-|-|-|
|BASE|IN04|IN05|IN07|IN08|MID|OUT0|OUT1|OUT2|OUT3|OUT4|OUT05|

SDXL - LyCORIS/LoCon(20)
|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|BASE|IN00|IN01|IN02|IN03|IN04|IN05|IN06|IN07|IN08||MID|OUT00|OUT01|OUT02|OUT03|OUT04|OUT05|OUT06|OUT07|OUT08|

### Special Values (Random)
Basically, a numerical value must be entered to work correctly, but by entering `R` and `U`, a random value will be entered.  
R : Numerical value with 3 decimal places from 0~1
U : 3 decimal places from -1.5 to 1.5

For example, if ROUT:1,1,1,1,1,1,1,1,R,R,R,R,R,R,R,R,R  
Only the OUT blocks is randomized.
The randomized values will be displayed on the command prompt screen when the image is generated.

### Special Values (Dynamic)
The special value `X` may also be included to use a dynamic weight specified in the LoRA syntax. This is activated by including an additional weight value after the specified `Original Weight`.

For example, if ROUT:X,1,1,1,1,1,1,1,1,1,1,1,X,X,X,X,X and you had a prompt containing \<lora:my_lore:0.5:ROUT:0.7\>. The `X` weights in ROUT would be replaced with `0.7` at runtime.

> NOTE: If you select an `Original Weight` tag that has a dynamic weight (`X`) and you do not specify a value in the LoRA syntax, it will default to `1`.

### Save Presets

The "Save Presets" button saves the text in the current text box. It is better to use a text editor, so use the "Open TextEditor" button to open a text editor, edit the text, and reload it.  
The text box above the Weights setting is a list of currently available identifiers, useful for copying and pasting into an XY plot. 17 identifiers are required to appear in the list.

### Fun Usage
Used in conjunction with the XY plot, it is possible to examine the impact of each level of the hierarchy.  
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
The optimal value can be searched by changing the value of each layer individually.
### Usage 
Check "Active" to activate the function. If Script (such as XYZ plot in Automatic1111) is enabled, it will take precedence.
Hires. fix is not supported. batch size is fixed to 1. batch count should be set to 1.  
Enter XYZ as the identifier of the LoRA that you want to change. It will work even if you do not enter a value corresponding to XYZ in the preset. If a value corresponding to XYZ is entered, that value will be used as the initial value.  
Inputting ZYX, inverted value will be automatically inputted.
This feature enables to match weights of two LoRAs.  
Inputing XYZ for LoRA1 and ZYX for LoRA2, you get,  
LoRA1 1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0  
LoRA2 0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1    
### Axis type
#### values
Sets the weight of the hierarchy to be changed. Enter the values separated by commas. 0,0.25,0.5,0.75,1", etc.

#### Block ID
If a block ID is entered, only that block will change to the value specified by value. As with the other types, use commas to separate them. Multiple blocks can be changed at the same time by separating them with a space or hyphen. The initial NOT will invert the change, so NOT IN09-OUT02 will change all blocks except IN09-OUT02.

#### seed
Seed changes, and is intended to be specified on the Z-axis.

#### Original Weights
Specify the initial value to change the weight of each block. If Original Weight is enabled, the value entered for XYZ is ignored.

### Input example
X : value, value : 1,0.25,0.5,0.75,1  
Y : Block ID, value : BASE,IN01-IN08,IN05-OUT05,OUT03-OUT11,NOT OUT03-OUT11  
Z : Original Weights, Value : NONE,ALL0.5,ALL  

In this case, an XY plot is created corresponding to the initial values NONE,ALL0.5,ALL.
If you select Seed for Z and enter -1,-1,-1, the XY plot will be created 3 times with different seeds.

### Original Weights Combined XY Plot
If both X and Y are set to Original Weights then an XY plot is made by combining the weights. If both X and Y have a weight in the same block then the Y case is set to zero before adding the arrays, this value will be used during the YX case where X's value is then set to zero. The intended usage is without overlapping blocks.

Given these names and values in the "Weights setting":  
INS:1,1,1,0,0,0,0,0,0,0,0,0  
MID:1,0,0,0,0,1,0,0,0,0,0,0  
OUTD:1,0,0,0,0,0,1,1,1,0,0,0  

With:  
X : Original Weights, value: INS,MID,OUTD  
Y : Original Weights, value: INS,MID,OUTD  
Z : none  

An XY plot is made with 9 elements. The diagonal is the X values: INS,MID,OUTD unchanged. So we have for the first row:
```
INS+INS  = 1,1,1,0,0,0,0,0,0,0,0,0 (Just INS unchanged, first image on the diagonal)
MID+INS  = 1,1,1,0,0,1,0,0,0,0,0,0 (second column of first row)
OUTD+INS = 1,1,1,0,0,0,1,1,1,0,0,0 (third column of first row)
```

Then the next row is INS+MID, MID+MID, OUTD+MID, and so on. Example image [here](https://user-images.githubusercontent.com/55250869/270830887-dff65f45-823a-4dbd-94c5-34d37c84a84f.jpg)

### Effective Block Analyzer
This function check which layers are working well. The effect of the block is visualized and quantified by setting the intensity of the other bocks to 1, decreasing the intensity of the block you want to examine, and taking the difference.  
#### Range
If you enter 0.5, 1, all initial values are set to 1, and only the target block is calculated as 0.5. Normally, 0.5 will make a difference, but some LoRAs may have difficulty making a difference, in which case, set 0.5 to 0 or a negative value.

#### settings
##### diff color
Specify the background color of the diff file.

##### chnage X-Y
Swaps the X and Y axes. By default, Block is assigned to the Y axis.

##### Threshold
Sets the threshold at which a change is recognized when calculating the difference. Basically, the default value is fine, but if you want to detect subtle differences in color, etc., lower the value.

#### Blocks
Enter the blocks to be examined, using the same format as for XYZ plots.

Here is the English translation in Markdown format:

### Guide for API users
#### Regular Usage
By default, Active is checked in the initial settings, so you can use it simply by installing it. You can use it by entering the format as instructed in the prompt. If executed, the phrase "LoRA Block Weight" will appear on the command prompt screen. If for some reason Active is not enabled, you can make it active by entering a value in the API for `"alwayson_scripts"`. 
When you enable API mode and use the UI, two extensions will appear. Please use the one on the bottom.
The default presets can be used for presets. If you want to use your own presets, you can either edit the preset file or use the following format for the data passed to the API. 

The code that can be used when passing to the API in json format is as follows. The presets you enter here will become available. If you want to use multiple presets, please separate them with `\n`.

```json
"prompt": "myprompt, <lora:mylora:1:MYSETS>",
"alwayson_scripts": {
    "LoRA Block Weight": {
        "args": ["MYSETS:1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\nYOURSETS:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1", true, 1 ,"","","","","","","","","","","","","",""]
    }
}
```
#### XYZ Plot
Please use the format below. Please delete `"alwayson_scripts"` as it will cause an error.

```json
"prompt": "myprompt, <lora:mylora:1:XYZ>",
"script_name":"LoRA Block Weight",
"script_args": ["XYZ:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1", true, 1 ,"seed","-1,-1","","","","","","","","","","","",""]
```
In this case, the six following `True,1` correspond to `xtype,xvalues,ytype,yvalues,ztype,zvalues`. It will be ignored if left blank. Please follow the instructions in the XYZ plot section for entering values. Even numbers should be enclosed in `""`.

The following types are available.

```json
"none","Block ID","values","seed","Original Weights","elements"
```
#### Effective Block Analyzer
It can be used by using the following format.

```json
"prompt": "myprompt, <lora:mylora:1:XYZ>",
"script_name":"LoRA Block Weight",
"script_args": ["XYZ:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1", true, 2 ,"","","","","","","0,1","17ALL",1,"white",20,true,"",""]
```
For `"0,1"`, specify the weight. If you specify `"17ALL"`, it will examine all the layers of the normal LoRA. If you want to specify individually, please write like `"BASE,IN00,IN01,IN02"`. Specify whether to reverse XY for `True` in the `"1"` for the number of times you want to check (if it is 2 or more, multiple seeds will be set), and `white` for the background color.

#### Make Weights
In "make weights," you can create a weight list from a slider. When you press the "add to preset" button, the weight specified by the identifier is added to the end of the preset. If a preset with the same name already exists, it will be overwritten. The "add to preset and save" button allows you to save the preset simultaneously.
![makeweights](https://github.com/hako-mikan/sd-webui-lora-block-weight/assets/122196982/9f0f3c1f-d824-45a6-926d-e1b431d5ef61)

# 概要
Loraは強力なツールですが、時に扱いが難しく、影響してほしくないところにまで影響がでたりします。このスクリプトではLoraを適用する際、適用度合いをU-Netの階層ごとに設定することができます。これを使用することで求める画像に近づけることができるかもしれません。

## 使い方
インストール時はWeb-ui.batを再起動をしてください。

### Active  
ここにチェックを入れることで動作します。

### プロンプト
プロンプト画面では通常通り使用したいLoraを記入してください。その際、強さの値の次に「:」を入力しウェイトか識別子を入力します。識別子はWeights setting で編集します。  
```
<lora:"lora name":1:0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0>.  
<lora:"lora name":1:0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0>.  (a1111-sd-webui-locon, etc.)
<lora:"lora name":1:1:lbw=IN02>  (a1111-sd-webui-lycoris, web-ui 1.5 or later)
<lora:"lora name":1:1:lbw=1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0>  (a1111-sd-webui-lycoris, web-ui 1.5 or later)
<lora:"lora name":1:1:lbw=IN02:stop=10>
```
Loraの強さは有効で、階層全体にかかります。大文字と小文字は区別されます。
LyCORISに対してLoRAのプリセットも使用できますが、その場合LoRAで使われていない階層のウェイトは0に設定されます。  
上記の形式になっていない場合プリセットではコメント行として扱われます。
a1111-sd-webui-lycoris版のLyCORISや、ver1.5以降のweb-uiを使用する場合構文が異なります。`lbw=IN02`を使って下さい。順番は問いません。その他の書式はlycorisの書式にしたがって下さい。詳しくはLyCORISのドキュメントを参照して下さい。識別子を入力して下さい。a1111-sd-webui-lycoris版は開発途中のためこの構文は変更される可能性があります。
### start, stop step  
`<lora:"lora name":lbw=ALL:start=10>`と指定すると、指定したstepからLoRAの効果が現れます。  
`<lora:"lora name":lbw=ALL:stop=10>`と指定することで指定したstepでLoRAの効果を無くします。キャラクターや構図LoRAの場合には10 step程度で十分な効果があり、ここで切ることで画風への影響を抑えることが可能です。  
`<lora:"lora name":lbw=ALL:step=5-10>`と指定するとstep 5-10の間のみLoRAが有効化します。

### Weights setting
識別子とウェイトを入力します。
フルモデルと異なり、Loraではエンコーダーを含め17のブロックに分かれています。よって、17個の数値を入力してください。
BASE,IN,OUTなどはフルモデル相当の階層です。
フルモデルやLyCORIS、SDXLなど様々な形式があるため、現状では12,17,20,26のウェイトを受け付けます。基本的に形式が合わないウェイトを入力しても動作しますが、未入力の層は0として扱われます。
|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|  
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|BASE|IN01|IN02|IN04|IN05|IN07|IN08|MID|OUT03|OUT04|OUT05|OUT06|OUT07|OUT08|OUT09|OUT10|OUT11|

LyCORISなどの場合(26)
|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|BASE|IN00|IN01|IN02|IN03|IN04|IN05|IN06|IN07|IN08|IN09|IN10|IN11|MID|OUT00|OUT01|OUT02|OUT03|OUT04|OUT05|OUT06|OUT07|OUT08|OUT09|OUT10|OUT11|

SDXL LoRAの場合(12)
|1|2|3|4|5|6|7|8|9|10|11|12|
|-|-|-|-|-|-|-|-|-|-|-|-|
|BASE|IN04|IN05|IN07|IN08|MID|OUT0|OUT1|OUT2|OUT3|OUT4|OUT05|

SDXL - LyCORISの場合(20)
|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|BASE|IN00|IN01|IN02|IN03|IN04|IN05|IN06|IN07|IN08||MID|OUT00|OUT01|OUT02|OUT03|OUT04|OUT05|OUT06|OUT07|OUT08|

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
ZYXと入力するとXYZとは反対の値が入力されます。これはふたつのLoRAのウェイトを合わせる際に有効です。
例えばLoRA1にXYZ,LoRA2にZYXと入力すると、  
LoRA1 1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0  
LoRA2 0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1    
となります。
### 軸タイプ
#### values
変化させる階層のウェイトを設定します。カンマ区切りで入力してください。「0,0.25,0.5,0.75,1」など。

#### Block ID
ブロックIDを入力すると、そのブロックのみvalueで指定した値に変わります。他のタイプと同様にカンマで区切ります。スペースまたはハイフンで区切ることで複数のブロックを同時に変化させることもできます。最初にNOTをつけることで変化対象が反転します。NOT IN09-OUT02とすると、IN09-OUT02以外が変化します。NOTは最初に入力しないと効果がありません。IN08-M00-OUT03は繋がっています。

#### Seed
シードが変わります。Z軸に指定することを想定しています。

#### Original Weights
各ブロックのウェイトを変化させる初期値を指定します。プリセットに登録されている識別子を入力してください。Original Weightが有効になっている場合XYZに入力された値は無視されます。

### Original Weightsの合算
もしXとYが両方ともOriginal Weightsに設定されている場合、その重みを組み合わせてXYプロットが作成されます。XとYの両方が同じブロックに重みがある場合、配列を加算する前にYケースはゼロに設定されます。この値は、Xの値がゼロに設定されるYXケースで使用されます。意図されている使用方法は、重複するブロックなしでのものです。

"Weights setting"に以下の名前と値が与えられているとします：  
INS:1,1,1,0,0,0,0,0,0,0,0,0  
MID:1,0,0,0,0,1,0,0,0,0,0,0  
OUTD:1,0,0,0,0,0,1,1,1,0,0,0  

以下の設定で：  
X : Original Weights, 値: INS,MID,OUTD  
Y : Original Weights, 値: INS,MID,OUTD  
Z : なし  

9つの要素を持つXYプロットが作成されます。対角線上は、変更されていないXの値：INS,MID,OUTDです。したがって、最初の行は以下のようになります：
```
INS+INS  = 1,1,1,0,0,0,0,0,0,0,0,0 (変更されていないINSだけ、対角線上の最初の画像)
MID+INS  = 1,1,1,0,0,1,0,0,0,0,0,0 (最初の行の第2列)
OUTD+INS = 1,1,1,0,0,0,1,1,1,0,0,0 (最初の行の第3列)
```

次の行は、INS+MID、MID+MID、OUTD+MIDなどです。例の画像は[こちら](https://user-images.githubusercontent.com/55250869/270830887-dff65f45-823a-4dbd-94c5-34d37c84a84f.jpg)です。

### 入力例
X : value, 値 : 1,0.25,0.5,0.75,1  
Y : Block ID, 値 : BASE,IN01-IN08,IN05-OUT05,OUT03-OUT11,NOT OUT03-OUT11  
Z : Original Weights, 値 : NONE,ALL0.5,ALL  

この場合、初期値NONE,ALL0.5,ALLに対応したXY plotが作製されます。
ZにSeedを選び、-1,-1,-1を入力すると、異なるseedでXY plotを3回作製します。

### Effective Block Analyzer
どの階層が良く効いているかを判別する機能です。対象の階層以外の強度を1にして、調べたい階層の強度を下げ、差分を取ることで階層の効果を可視化・数値化します。  
#### Range
0.5, 1　と入力した場合、初期値がすべて1になり、対象のブロックのみ0.5として計算が行われます。普通は0.5で差がでますが、LoRAによっては差が出にくい場合があるので、その場合は0.5を0あるいはマイナスの値に設定してください。

#### 設定
##### diff color
差分ファイルの背景カラーを指定します。

##### chnage X-Y
X軸とY軸を入れ替えます。デフォルトではY軸にBlockが割り当てられています。

##### Threshold
差分を計算する際の変化したと認識される閾値を設定します。基本的にはデフォルト値で問題ありませんが、微妙な色の違いなどを検出したい場合は値を下げて下さい。

#### Blocks
調べたい階層を入力します。XYZプロットと同じ書式が使用可能です。

階層別マージについては下記を参照してください

### elemental
詳細は[こちら](https://github.com/hako-mikan/sd-webui-supermerger/blob/main/elemental_ja.md)を参照して下さい。
#### 使い方
Elementaタブにて階層指定と同じように識別子を設定します。識別子は階層の識別子の後に入力します。
\<lora:"lora名":1:IN04:ATTNON>
ATTNON:

書式は  
識別子:階層指定:要素指定:ウェイト  
のように指定します。要素は部分一致で判定されます。attn1ならattn1のみ、attnならattn1及びattn2が反応します。階層、要素共に空白で区切ると複数指定できます。  
print changeをオンにすると反応した要素がコマンドプロンプト上に表示されます。

ALL0:::0  
はすべての要素のウェイトをゼロに設定します。  
IN1:IN00-IN11::1  
はINのすべての要素を1にします  
ATTNON::attn:1
はすべての階層のattnを1にします。

#### XYZプロット
XYZプロットのelementsの項ではカンマ区切りでXYZプロットが可能になります。
その場合は  
\<lora:"lora名":1:XYZ:XYZ>  
と指定して下さい。
elements  
の項に  
IN05-OUT05:attn:0,IN05-OUT05:attn:0.5,IN05-OUT05:attn:1  
と入力して走らせるとIN05からOUT05までのattnのみを変化させることができます。
この際、XYZの値を変更することで初期値を変更できます。デフォルトではelementalのXYZはXYZ:::1となっており、これは全階層、全要素を1にしますが、ここをXYZ:encoder::1とするとテキストエンコーダーのみを有効にした状態で評価ができます。

#### ウェイトの作成
make weightsではスライダーからウェイトリストを作成できます。
add to presetボタンを押すと、identiferで指定されたウェイトがプリセットの末尾に追加されます。
すでに同じ名前のプリセットが存在する場合、上書きされます。
add to preset and saveボタンでは同時にプリセットの保存が行われます。
![makeweights](https://github.com/hako-mikan/sd-webui-lora-block-weight/assets/122196982/9f0f3c1f-d824-45a6-926d-e1b431d5ef61)

### APIを通しての利用について
#### 通常利用
初期設定でActiveはチェックされているのでインストールするだけで利用可能になります。
プロンプトに書式通りに入力することで利用できます。実行された場合にはコマンドプロンプト画面に「LoRA Block Weight」の文字が現れます。
何らかの理由でActiveになっていない場合にはAPIに投げる値のうち、`"alwayson_scripts"`に値を入力することでActiveにできます。
APIモードを有効にしてUIを使うとき、拡張がふたつ表示されます。下の方を使って下さい。
プリセットはデフォルトのプリセットが利用できます。独自のプリセットを利用したい場合にはプリセットファイルを編集するか、APIに受け渡すデータに対して下記の書式を利用して下さい。
json形式でAPIに受け渡すときに使用できるコードです。ここで入力したプリセットが利用可能になります。複数のプリセットを利用したい場合には`\n`で区切って下さい。

    "prompt": "myprompt, <lora:mylora:1:MYSETS>",
	"alwayson_scripts": {
		"LoRA Block Weight": {
			"args": ["MYSETS:1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\nYOURSETS:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1", True, 1 ,"","","","","","","","","","","","","",""]
		}
    }

#### XYZ plot
下記の書式を利用して下さい。`"alwayson_scripts"`は消して下さいエラーになります。
```
    "prompt": "myprompt, <lora:mylora:1:XYZ>",
    "script_name":"LoRA Block Weight",
    "script_args": ["XYZ:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1", True, 1 ,"seed","-1,-1","","","","","","","","","","","",""]

```
この際、`True,1`に続く6個が`xtype,xvalues,ytype,yvalues,ztype,zvalues`に対応します。空白だと無視されます。入力する値などはXYZ plotの項に従って下さい。数字でもすべて`""`で囲って下さい。
使用できるタイプは次の通りです。
```
"none","Block ID","values","seed","Original Weights","elements"
```
#### Effective Block Analyzer
下記のような書式を使うことで使用できます。
```
    "prompt": "myprompt, <lora:mylora:1:XYZ>",
    "script_name":"LoRA Block Weight",
 "script_args": ["XYZ:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1", True, 2 ,"","","","","","","0,1","17ALL",1,"white",20,True,"",""]

```
`"0,1"`にはウェイト。`"17ALL"`を指定すると普通のLoRAすべての階層を調べます。個別に指定したい場合は`"BASE,IN00,IN01,IN02"`のように記述して下さい。`1`には調べたい回数(2以上だと複数のシードを設定します),`white`には背景色,`True`にはXYを反転するかどうかを指定して下さい。

### updates/更新情報
### 2023.10.26.2000(JST)
- bugfix:Effective block checker does not work correctly.
- bugfix:Does not work correctly when lora in memory is set to a value other than 0.

### 2023.10.04.2000(JST)  
XYZ plotに[新たな機能](#Original-Weightsの合算)が追加されました。[sometimesacoder](https://github.com/sometimesacoder)氏に感謝します。  
A [new feature](#Original-Weights-Combined-XY-Plot) was added to the XYZ plot. Many thanks to [sometimesacoder](https://github.com/sometimesacoder).

### 2023.07.22.0030(JST)
- support SDXL
- support web-ui 1.5
- support no buildin-LoRA system(lycoris required)

to use with web-ui 1.5/web-ui1.5で使うときは
```
<lora:"lora name":1:1:lbw=IN02>  
<lora:"lora name":1:1:lbw=1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0>  
```

### 2023.07.14.2000(JST)
- APIでXYZプロットが利用可能になりました
- [APIの利用方法](#apiを通しての利用について)を追記しました
- XYZ plot can be used in API
- Added [guide for API users](#guide-for-api-users)

### 2023.5.24.2000(JST)
- changed directory for presets(extentions/sd-webui-lora-block-weight/scripts/)
- プリセットの保存フォルダがextentions/sd-webui-lora-block-weight/scripts/に変更になりました。

### 2023.5.12.2100(JST)
- changed syntax of lycoris
- lycorisの書式を変更しました

### 2023.04.14.2000(JST)
- support LyCORIS(a1111-sd-webui-lycoris)
- LyCORIS(a1111-sd-webui-lycoris)に対応

### 2023.03.20.2030(JST)
- Comment lines can now be added to presets
- プリセットにコメント行を追加できるようになりました
- support XYZ plot hires.fix
- XYZプロットがhires.fixに対応しました

### 2023.03.16.2030(JST)
- [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS)に対応しました
- Support [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS)

別途[LyCORIS Extention](https://github.com/KohakuBlueleaf/a1111-sd-webui-locon)が必要です。
For use LyCORIS, [Extension](https://github.com/KohakuBlueleaf/a1111-sd-webui-locon) for LyCORIS needed.

### 2023.02.07 1250(JST)
- Changed behavior when XYZ plot Active (Script of the main UI is prioritized).

### 2023.02.06 2000(JST)
- Feature added: XYZ plotting is added.

### 2023.01.31 0200(JST)
- Feature added: Random feature is added
- Fixed: Weighting now works for negative values.

### 2023.02.16 2040(JST)
- Original Weight をxやyに設定できない問題を解決しました
- Effective Weight Analyzer選択時にXYZのXやYがValuesとBlockIdになっていないとエラーになる問題を解決しました

### 2023.02.08 2120(JST)
- 階層適応した後通常使用する際、階層適応が残る問題を解決しました
- 効果のある階層をワンクリックで判別する機能を追加しました

### 2023.02.08 0050(JST)
- 一部環境でseedが固定されない問題を解決しました

### 2023.02.07 2015(JST)
- マイナスのウェイトが正常に働かない問題を修正しました

### 2023.02.07 1250(JST)
- XYZプロットActive時の動作を変更しました(本体のScriptが優先されるようになります)

### 2023.02.06 2000(JST)
- 機能追加：XYZプロット機能を追加しました

### 2023.01.31 0200(JST)
- 機能追加：ランダム機能を追加しました
- 機能修正：ウェイトがマイナスにも効くようになりました
