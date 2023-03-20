import cv2
import os
import re
import torch
import shutil
import math
import numpy as np
import gradio as gr
import os.path
import random
from pprint import pprint
import modules.ui
import modules.scripts as scripts
from PIL import Image, ImageFont, ImageDraw
from fonts.ttf import Roboto
import modules.shared as shared
from modules import devices, sd_models, images,extra_networks
from modules.shared import opts, state
from modules.processing import process_images, Processed

lxyz = ""
lzyx = ""

BLOCKS=["encoder",
"diffusion_model_input_blocks_0_",
"diffusion_model_input_blocks_1_",
"diffusion_model_input_blocks_2_",
"diffusion_model_input_blocks_3_",
"diffusion_model_input_blocks_4_",
"diffusion_model_input_blocks_5_",
"diffusion_model_input_blocks_6_",
"diffusion_model_input_blocks_7_",
"diffusion_model_input_blocks_8_",
"diffusion_model_input_blocks_9_",
"diffusion_model_input_blocks_10_",
"diffusion_model_input_blocks_11_",
"diffusion_model_middle_block_",
"diffusion_model_output_blocks_0_",
"diffusion_model_output_blocks_1_",
"diffusion_model_output_blocks_2_",
"diffusion_model_output_blocks_3_",
"diffusion_model_output_blocks_4_",
"diffusion_model_output_blocks_5_",
"diffusion_model_output_blocks_6_",
"diffusion_model_output_blocks_7_",
"diffusion_model_output_blocks_8_",
"diffusion_model_output_blocks_9_",
"diffusion_model_output_blocks_10_",
"diffusion_model_output_blocks_11_"]

loopstopper = True

ATYPES =["none","Block ID","values","seed","Original Weights"]

class Script(modules.scripts.Script):   
    def title(self):
        return "LoRA Block Weight"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    def ui(self, is_img2img):
        import lora
        LWEIGHTSPRESETS="\
NONE:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n\
ALL:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1\n\
INS:1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0\n\
IND:1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0\n\
INALL:1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0\n\
MIDD:1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0\n\
OUTD:1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0\n\
OUTS:1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1\n\
OUTALL:1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1\n\
ALL0.5:0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5"

        runorigin = scripts.scripts_txt2img.run
        runorigini = scripts.scripts_img2img.run

        path_root = scripts.basedir()
        extpath = os.path.join(path_root,"extensions","sd-webui-lora-block-weight","scripts", "lbwpresets.txt")
        filepath = os.path.join(path_root,"scripts", "lbwpresets.txt")

        if os.path.isfile(extpath) and not os.path.isfile(filepath):
            shutil.move(extpath,filepath)
            
        lbwpresets=""

        try:
            with open(filepath) as f:
                lbwpresets = f.read()
        except OSError as e:
                lbwpresets=LWEIGHTSPRESETS
                if not os.path.isfile(filepath):
                    try:
                        with open(filepath,mode = 'w') as f:
                            f.write(lbwpresets)
                    except:
                        pass

        loraratios=lbwpresets.splitlines()
        lratios={}
        for i,l in enumerate(loraratios):
            if ":" not in l or not (l.count(",") == 16 or l.count(",") == 25) : continue
            lratios[l.split(":")[0]]=l.split(":")[1]
        ratiostags = [k for k in lratios.keys()]
        ratiostags = ",".join(ratiostags)

        with gr.Accordion("LoRA Block Weight",open = False):
            with gr.Row():
                with gr.Column(min_width = 50, scale=1):
                    lbw_useblocks =  gr.Checkbox(value = True,label="Active",interactive =True,elem_id="lbw_active")
                with gr.Column(scale=5):
                    bw_ratiotags= gr.TextArea(label="",lines=2,value=ratiostags,visible =True,interactive =True,elem_id="lbw_ratios") 
            with gr.Accordion("XYZ plot",open = False):
                gr.HTML(value="<p>changeable blocks : BASE,IN00,IN01,IN02,IN03,IN04,IN05,IN06,IN07,IN08,IN09,IN10,IN11,M00,OUT00,OUT01,OUT02,OUT03,OUT04,OUT05,OUT06,OUT07,OUT08,OUT09,OUT10,OUT11</p>")
                xyzsetting = gr.Radio(label = "Active",choices = ["Disable","XYZ plot","Effective Block Analyzer"], value ="Disable",type = "index") 
                with gr.Row(visible = False) as esets:
                    diffcol = gr.Radio(label = "diff image color",choices = ["black","white"], value ="black",type = "value",interactive =True) 
                    revxy = gr.Checkbox(value = False,label="change X-Y",interactive =True,elem_id="lbw_changexy")
                    thresh = gr.Textbox(label="difference threshold",lines=1,value="20",interactive =True,elem_id="diff_thr")
                xtype = gr.Dropdown(label="X Types         ", choices=[x for x in ATYPES], value=ATYPES [2],interactive =True,elem_id="lbw_xtype")
                xmen = gr.Textbox(label="X Values         ",lines=1,value="0,0.25,0.5,0.75,1",interactive =True,elem_id="lbw_xmen")
                ytype = gr.Dropdown(label="Y Types         ", choices=[y for y in ATYPES], value=ATYPES [1],interactive =True,elem_id="lbw_ytype")    
                ymen = gr.Textbox(label="Y Values         " ,lines=1,value="IN05-OUT05",interactive =True,elem_id="lbw_ymen")
                ztype = gr.Dropdown(label="Z type         ", choices=[z for z in ATYPES], value=ATYPES[0],interactive =True,elem_id="lbw_ztype")    
                zmen = gr.Textbox(label="Z values         ",lines=1,value="",interactive =True,elem_id="lbw_zmen")

                exmen = gr.Textbox(label="Range",lines=1,value="0.5,1",interactive =True,elem_id="lbw_exmen",visible = False) 
                eymen = gr.Textbox(label="Blocks" ,lines=1,value="BASE,IN00,IN01,IN02,IN03,IN04,IN05,IN06,IN07,IN08,IN09,IN10,IN11,M00,OUT00,OUT01,OUT02,OUT03,OUT04,OUT05,OUT06,OUT07,OUT08,OUT09,OUT10,OUT11",interactive =True,elem_id="lbw_eymen",visible = False)  

            with gr.Accordion("Weights setting",open = True):
                with gr.Row():
                    reloadtext = gr.Button(value="Reload Presets",variant='primary',elem_id="lbw_reload")
                    reloadtags = gr.Button(value="Reload Tags",variant='primary',elem_id="lbw_reload")
                    savetext = gr.Button(value="Save Presets",variant='primary',elem_id="lbw_savetext")
                    openeditor = gr.Button(value="Open TextEditor",variant='primary',elem_id="lbw_openeditor")
                lbw_loraratios = gr.TextArea(label="",value=lbwpresets,visible =True,interactive  = True,elem_id="lbw_ratiospreset")      
        
        import subprocess
        def openeditors():
            subprocess.Popen(['start', filepath], shell=True)
        
        def reloadpresets():
            try:
                with open(filepath) as f:
                    return f.read()
            except OSError as e:
                pass

        def tagdicter(presets):
            presets=presets.splitlines()
            wdict={}
            for l in presets:
                if ":" not in l or not (l.count(",") == 16 or l.count(",") == 25) : continue
                w=[]
                if ":" in l :
                    key = l.split(":",1)[0]
                    w = l.split(":",1)[1]
                if len([w for w in w.split(",")]) == 17 or len([w for w in w.split(",")]) ==26:
                    wdict[key.strip()]=w
            return ",".join(list(wdict.keys()))

        def savepresets(text):
            with open(filepath,mode = 'w') as f:
                f.write(text)

        reloadtext.click(fn=reloadpresets,inputs=[],outputs=[lbw_loraratios])
        reloadtags.click(fn=tagdicter,inputs=[lbw_loraratios],outputs=[bw_ratiotags])
        savetext.click(fn=savepresets,inputs=[lbw_loraratios],outputs=[])
        openeditor.click(fn=openeditors,inputs=[],outputs=[])


        def urawaza(active):
            if active > 0:
                for obj in scripts.scripts_txt2img.alwayson_scripts:
                    if "lora_block_weight" in obj.filename:
                        scripts.scripts_txt2img.selectable_scripts.append(obj)
                        scripts.scripts_txt2img.titles.append("LoRA Block Weight")
                for obj in scripts.scripts_img2img.alwayson_scripts:
                    if "lora_block_weight" in obj.filename:
                        scripts.scripts_img2img.selectable_scripts.append(obj)
                        scripts.scripts_img2img.titles.append("LoRA Block Weight")
                scripts.scripts_txt2img.run = newrun
                scripts.scripts_img2img.run = newrun
                if active == 1:return [*[gr.update(visible = True) for x in range(6)],*[gr.update(visible = False) for x in range(3)]]
                else:return [*[gr.update(visible = False) for x in range(6)],*[gr.update(visible = True) for x in range(3)]]
            else:
                scripts.scripts_txt2img.run = runorigin
                scripts.scripts_img2img.run = runorigini
                return [*[gr.update(visible = True) for x in range(6)],*[gr.update(visible = False) for x in range(3)]]

        xyzsetting.change(fn=urawaza,inputs=[xyzsetting],outputs =[xtype,xmen,ytype,ymen,ztype,zmen,exmen,eymen,esets])

        return lbw_loraratios,lbw_useblocks,xyzsetting,xtype,xmen,ytype,ymen,ztype,zmen,exmen,eymen,diffcol,thresh,revxy

    def process(self, p, loraratios,useblocks,xyzsetting,xtype,xmen,ytype,ymen,ztype,zmen,exmen,eymen,diffcol,thresh,revxy):
        #print("self =",self,"p =",p,"presets =",loraratios,"useblocks =",useblocks,"xyzsettings =",xyzsetting,"xtype =",xtype,"xmen =",xmen,"ytype =",ytype,"ymen =",ymen,"ztype =",ztype,"zmen =",zmen)
        
        if useblocks:
            loraratios=loraratios.splitlines()
            lratios={}
            for l in loraratios:
                if ":" not in l or not (l.count(",") == 16 or l.count(",") == 25) : continue
                l0=l.split(":",1)[0]
                lratios[l0.strip()]=l.split(":",1)[1]
            if xyzsetting and "XYZ" in p.prompt:
                lratios["XYZ"] = lxyz
                lratios["ZYX"] = lzyx
            loradealer(p,lratios)
        return

    def postprocess(self, p, processed, *args):
        import lora
        lora.loaded_loras.clear()

    def run(self,p,presets,useblocks,xyzsetting,xtype,xmen,ytype,ymen,ztype,zmen,exmen,eymen,diffcol,thresh,revxy):
        if xyzsetting >0:
            import lora
            loraratios=presets.splitlines()
            lratios={}
            for l in loraratios:
                if ":" not in l or not (l.count(",") == 16 or l.count(",") == 25) : continue
                l0=l.split(":",1)[0]
                lratios[l0.strip()]=l.split(":",1)[1]

            if "XYZ" in p.prompt:
                base = lratios["XYZ"] if "XYZ" in lratios.keys() else "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
            else: return

            if xyzsetting > 1: 
                xmen,ymen = exmen,eymen
                xtype,ytype = "values","ID"
                ebase = xmen.split(",")[1]
                ebase = [ebase.strip()]*26
                base = ",".join(ebase)
                ztype = ""

            #ATYPES =["none","Block ID","values","seed","Base Weights"]

            def dicedealer(am):
                for i,a in enumerate(am):
                    if a =="-1": am[i] = str(random.randrange(4294967294))
                print(f"the die was thrown : {am}")

            if p.seed == -1: p.seed = str(random.randrange(4294967294))
                
            #print(f"xs:{xmen},ys:{ymen},zs:{zmen}")

            def adjuster(a,at):
                if "none" in at:a = ""
                a = [a.strip() for a in a.split(',')]
                if "seed" in at:dicedealer(a)
                return a
            
            xs = adjuster(xmen,xtype)
            ys = adjuster(ymen,ytype)
            zs = adjuster(zmen,ztype)

            ids = alpha =seed = ""
            p.batch_size = 1

            print(f"xs:{xs},ys:{ys},zs:{zs}")

            images = []

            def weightsdealer(alpha,ids,base):
                blockid17=["BASE","IN01","IN02","IN04","IN05","IN07","IN08","M00","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
                blockid26=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
                #print(f"weights from : {base}")
                ids = [z.strip() for z in ids.split(' ')]
                weights_t = [w.strip() for w in base.split(',')]
                blockid = blockid17 if len(weights_t) ==17 else blockid26
                if ids[0]!="NOT":
                    flagger=[False]*len(weights_t)
                    changer = True
                else:
                    flagger=[True]*len(weights_t)
                    changer = False
                for id in ids:
                    if id =="NOT":continue
                    if "-" in id:
                        it = [it.strip() for it in id.split('-')]
                        if  blockid.index(it[1]) > blockid.index(it[0]):
                            flagger[blockid.index(it[0]):blockid.index(it[1])+1] = [changer]*(blockid.index(it[1])-blockid.index(it[0])+1)
                        else:
                            flagger[blockid.index(it[1]):blockid.index(it[0])+1] = [changer]*(blockid.index(it[0])-blockid.index(it[1])+1)
                    else:
                        flagger[blockid.index(id)] =changer    
                for i,f in enumerate(flagger):
                    if f:weights_t[i]=alpha
                outext = ",".join(weights_t)
                #print(f"weights changed: {outext}")
                return outext

            def xyzdealer(a,at):
                nonlocal ids,alpha,p,base,c_base
                if "ID" in at:return
                if "values" in at:alpha = a
                if "seed" in at:
                    p.seed = int(a)
                if "Weights" in at:base =c_base = lratios[a]

            grids = []
            images =[]

            totalcount = len(xs)*len(ys)*len(zs) if xyzsetting < 2 else len(xs)*len(ys)*len(zs)  //2 +1
            shared.total_tqdm.updateTotal(totalcount)
            xc = yc =zc = 0
            state.job_count = totalcount 
            totalcount = len(xs)*len(ys)*len(zs)

            for z in zs:
                images = []
                yc = 0
                xyzdealer(z,ztype)
                for y in ys:
                    xc = 0
                    xyzdealer(y,ytype)
                    for x in xs:
                        xyzdealer(x,xtype)
                        if "ID" in xtype:
                            if "values" in ytype:c_base = weightsdealer(y,x,base)
                            if "values" in ztype:c_base = weightsdealer(z,x,base)
                        if "ID" in ytype:
                            if "values" in xtype:c_base = weightsdealer(x,y,base)
                            if "values" in ztype:c_base = weightsdealer(z,y,base)
                        if "ID" in ztype:
                            if "values" in xtype:c_base = weightsdealer(x,z,base)
                            if "values" in ytype:c_base = weightsdealer(y,z,base)

                        print(f"X:{xtype}, {x},Y: {ytype},{y}, Z:{ztype},{z}, base:{c_base} ({len(xs)*len(ys)*zc + yc*len(xs) +xc +1}/{totalcount})")
                        
                        global lxyz,lzyx
                        lxyz = c_base

                        cr_base = c_base.split(",")
                        cr_base_t=[]
                        for x in cr_base:
                            if not identifier(x):
                                cr_base_t.append(str(1-float(x)))
                            else:
                                cr_base_t.append(x)
                        lzyx = ",".join(cr_base_t)

                        if not(xc == 1 and not (yc ==0 ) and xyzsetting >1):
                            lora.loaded_loras.clear()
                            processed:Processed = process_images(p)
                            images.append(processed.images[0])
                        xc += 1
                    yc += 1
                zc += 1
                origin = loranames(processed.all_prompts) + ", "+ znamer(ztype,z,base)
                if xyzsetting >1: images,xs,ys = effectivechecker(images,xs,ys,diffcol,thresh,revxy)
                grids.append(smakegrid(images,xs,ys,origin,p))
            processed.images= grids
            lora.loaded_loras.clear()
            return processed

def identifier(char):
    return char[0] in ["R", "U", "X"]

def znamer(at,a,base):
    if "ID" in at:return f"Block : {a}"
    if "values" in at:return f"value : {a}"
    if "seed" in at:return f"seed : {a}"
    if "Weights" in at:return f"original weights :\n {base}"
    else: return ""

def loranames(all_prompts):
    _, extra_network_data = extra_networks.parse_prompts(all_prompts[0:1])
    calledloras = extra_network_data["lora"]
    names = ""
    for called in calledloras:
        if len(called.items) <3:continue
        names += called.items[0] 
    return names

def loradealer(p,lratios):
    _, extra_network_data = extra_networks.parse_prompts(p.all_prompts[0:1])
    calledloras = extra_network_data["lora"]
    lorans = []
    lorars = []
    for called in calledloras:
        if len(called.items) <3:continue
        if called.items[2] in lratios or called.items[2].count(",") ==16 or called.items[2].count(",") ==25:
            lorans.append(called.items[0])
            wei = lratios[called.items[2]] if called.items[2] in lratios else called.items[2] 
            multiple = called.items[1]
            ratios = [w.strip() for w in wei.split(",")]
            for i,r in enumerate(ratios):
                if r =="R":
                    ratios[i] = round(random.random(),3)
                elif r == "U":
                    ratios[i] = round(random.uniform(-0.5,1.5),3)
                elif r[0] == "X":
                    ratios[i] = getinheritedweight(multiple, r)
                else:
                    ratios[i] = float(r)
            print(f"LoRA Block weight :{called.items[0]}: {ratios}")
            if len(ratios)==17:
                ratios = [ratios[0]] + [1] + ratios[1:3]+ [1] + ratios[3:5]+[1] + ratios[5:7]+[1,1,1] + [ratios[7]] + [1,1,1] + ratios[8:]
            lorars.append(ratios)
    if len(lorars) > 0: load_loras_blocks(lorans,lorars,multiple)

re_digits = re.compile(r"\d+")

re_unet_down_blocks = re.compile(r"lora_unet_down_blocks_(\d+)_attentions_(\d+)_(.+)")
re_unet_mid_blocks = re.compile(r"lora_unet_mid_block_attentions_(\d+)_(.+)")
re_unet_up_blocks = re.compile(r"lora_unet_up_blocks_(\d+)_attentions_(\d+)_(.+)")

re_unet_down_blocks_res = re.compile(r"lora_unet_down_blocks_(\d+)_resnets_(\d+)_(.+)")
re_unet_mid_blocks_res = re.compile(r"lora_unet_mid_block_resnets_(\d+)_(.+)")
re_unet_up_blocks_res = re.compile(r"lora_unet_up_blocks_(\d+)_resnets_(\d+)_(.+)")

re_unet_downsample = re.compile(r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv(.+)")
re_unet_upsample = re.compile(r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv(.+)")

re_text_block = re.compile(r"lora_te_text_model_encoder_layers_(\d+)_(.+)")

re_inherited_weight = re.compile(r"X([+-])?([\d.]+)?")

def getinheritedweight(weight, offset):
    match = re_inherited_weight.search(offset)
    if match.group(1) == "+":
        return float(weight) + float(match.group(2))
    elif match.group(1) == "-":
        return float(weight) - float(match.group(2))  
    else:
        return float(weight) 

def convert_diffusers_name_to_compvis(key):
    def match(match_list, regex):
        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, re_unet_down_blocks):
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[1]}_1_{m[2]}"

    if match(m, re_unet_mid_blocks):
        return f"diffusion_model_middle_block_1_{m[1]}"

    if match(m, re_unet_up_blocks):
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[1]}_1_{m[2]}"

    if match(m, re_unet_down_blocks_res):
        block = f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[1]}_0_"
        if m[2].startswith('conv1'):
            return f"{block}in_layers_2{m[2][len('conv1'):]}"
        elif m[2].startswith('conv2'):
            return f"{block}out_layers_3{m[2][len('conv2'):]}"
        elif m[2].startswith('time_emb_proj'):
            return f"{block}emb_layers_1{m[2][len('time_emb_proj'):]}"
        elif m[2].startswith('conv_shortcut'):
            return f"{block}skip_connection{m[2][len('conv_shortcut'):]}"

    if match(m, re_unet_mid_blocks_res):
        block = f"diffusion_model_middle_block_{m[0]*2}_"
        if m[1].startswith('conv1'):
            return f"{block}in_layers_2{m[1][len('conv1'):]}"
        elif m[1].startswith('conv2'):
            return f"{block}out_layers_3{m[1][len('conv2'):]}"
        elif m[1].startswith('time_emb_proj'):
            return f"{block}emb_layers_1{m[1][len('time_emb_proj'):]}"
        elif m[1].startswith('conv_shortcut'):
            return f"{block}skip_connection{m[1][len('conv_shortcut'):]}"

    if match(m, re_unet_up_blocks_res):
        block = f"diffusion_model_output_blocks_{m[0] * 3 + m[1]}_0_"
        if m[2].startswith('conv1'):
            return f"{block}in_layers_2{m[2][len('conv1'):]}"
        elif m[2].startswith('conv2'):
            return f"{block}out_layers_3{m[2][len('conv2'):]}"
        elif m[2].startswith('time_emb_proj'):
            return f"{block}emb_layers_1{m[2][len('time_emb_proj'):]}"
        elif m[2].startswith('conv_shortcut'):
            return f"{block}skip_connection{m[2][len('conv_shortcut'):]}"

    if match(m, re_unet_downsample):
        return f"diffusion_model_input_blocks_{m[0]*3+3}_0_op{m[1]}"

    if match(m, re_unet_upsample):
        return f"diffusion_model_output_blocks_{m[0]*3 + 2}_{1+(m[0]!=0)}_conv{m[1]}"

    if match(m, re_text_block):
        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    return key

class FakeModule(torch.nn.Module):
    def __init__(self, weight, func):
        super().__init__()
        self.weight = weight
        self.func = func
    
    def forward(self, x):
        return self.func(x)

class LoraUpDownModule:
    def __init__(self):
        self.up_model = None
        self.mid_model = None
        self.down_model = None
        self.alpha = None
        self.dim = None
        self.op = None
        self.extra_args = {}
        self.shape = None
        self.bias = None
        self.up = None
    
    def down(self, x):
        return x
    
    def inference(self, x):
        if hasattr(self, 'bias') and isinstance(self.bias, torch.Tensor):
            out_dim = self.up_model.weight.size(0)
            rank = self.down_model.weight.size(0)
            rebuild_weight = (
                self.up_model.weight.reshape(out_dim, -1) @ self.down_model.weight.reshape(rank, -1)
                + self.bias
            ).reshape(self.shape)
            return self.op(
                x, rebuild_weight,
                **self.extra_args
            )
        else:
            if self.mid_model is None:
                return self.up_model(self.down_model(x))
            else:
                return self.up_model(self.mid_model(self.down_model(x)))

def pro3(t, wa, wb):
    temp = torch.einsum('i j k l, j r -> i r k l', t, wb)
    return torch.einsum('i j k l, i r -> r j k l', temp, wa)

class LoraHadaModule:
    def __init__(self):
        self.t1 = None
        self.w1a = None
        self.w1b = None
        self.t2 = None
        self.w2a = None
        self.w2b = None
        self.alpha = None
        self.dim = None
        self.op = None
        self.extra_args = {}
        self.shape = None
        self.bias = None
        self.up = None
    
    def down(self, x):
        return x
    
    def inference(self, x):
        if hasattr(self, 'bias') and isinstance(self.bias, torch.Tensor):
            bias = self.bias
        else:
            bias = 0
        
        if self.t1 is None:
            return self.op(
                x,
                ((self.w1a @ self.w1b) * (self.w2a @ self.w2b) + bias).view(self.shape),
                **self.extra_args
            )
        else:
            return self.op(
                x,
                (pro3(self.t1, self.w1a, self.w1b) 
                 * pro3(self.t2, self.w2a, self.w2b) + bias).view(self.shape),
                **self.extra_args
            )

CON_KEY = {
    "lora_up.weight",
    "lora_down.weight",
    "lora_mid.weight"
}
HADA_KEY = {
    "hada_t1",
    "hada_w1_a",
    "hada_w1_b",
    "hada_t2",
    "hada_w2_a",
    "hada_w2_b",
}

def load_lora(name, filename,lwei):
    import lora as lora_o
    lora = lora_o.LoraModule(name)
    lora.mtime = os.path.getmtime(filename)

    sd = sd_models.read_state_dict(filename)

    keys_failed_to_match = []
    keys_failed_to_match_lbw = []

    for key_diffusers, weight in sd.items():
        ratio = 1
        picked = False
        
        fullkey = convert_diffusers_name_to_compvis(key_diffusers)
        key, lora_key = fullkey.split(".", 1)

        for i,block in enumerate(BLOCKS):
            if block in key:
                ratio = lwei[i]
                picked = True
        
        if not picked:keys_failed_to_match_lbw.append(key_diffusers)

        weight = weight * math.sqrt(abs(ratio))
        
        sd_module = shared.sd_model.lora_layer_mapping.get(key, None)
        if sd_module is None:
            keys_failed_to_match.append(key_diffusers)
            continue

        lora_module = lora.modules.get(key, None)
        if lora_module is None:
            lora_module = LoraUpDownModule()
            lora.modules[key] = lora_module

        if lora_key == "alpha":
            lora_module.alpha = weight.item()
            continue
        
        if 'bias_' in lora_key:
            if lora_module.bias is None:
                lora_module.bias = [None, None, None]
            if 'bias_indices' == lora_key:
                lora_module.bias[0] = weight
            elif 'bias_values' == lora_key:
                lora_module.bias[1] = weight
            elif 'bias_size' == lora_key:
                lora_module.bias[2] = weight
            
            if all((i is not None) for i in lora_module.bias):
                print('build bias')
                lora_module.bias = torch.sparse_coo_tensor(
                    lora_module.bias[0],
                    lora_module.bias[1],
                    tuple(lora_module.bias[2]),
                ).to(device=devices.device, dtype=devices.dtype)
                lora_module.bias.requires_grad_(False)
            continue
        
        if lora_key in CON_KEY:
            if type(sd_module) == torch.nn.Linear:
                weight = weight.reshape(weight.shape[0], -1)
                module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
                lora_module.op = torch.nn.functional.linear
            elif type(sd_module) == torch.nn.Conv2d:
                if lora_key == "lora_down.weight":
                    if weight.shape[2] != 1 or weight.shape[3] != 1:
                        module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], sd_module.kernel_size, sd_module.stride, sd_module.padding, bias=False)
                    else:
                        module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
                elif lora_key == "lora_mid.weight":
                    module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], sd_module.kernel_size, sd_module.stride, sd_module.padding, bias=False)
                elif lora_key == "lora_up.weight":
                    module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
                lora_module.op = torch.nn.functional.conv2d
                lora_module.extra_args = {
                    'stride': sd_module.stride,
                    'padding': sd_module.padding
                }
            else:
                assert False, f'Lora layer {key_diffusers} matched a layer with unsupported type: {type(sd_module).__name__}'
            
            lora_module.shape = sd_module.weight.shape

            fugou = np.sign(ratio) if lora_key == "lora_up.weight" else 1

            with torch.no_grad():
                module.weight.copy_(weight*fugou)

            module.to(device=devices.device, dtype=devices.dtype)
            module.requires_grad_(False)

            if lora_key == "lora_up.weight":
                lora_module.up_model = module
                lora_module.up = FakeModule(
                    lora_module.up_model.weight,
                    lora_module.inference
                )
            elif lora_key == "lora_mid.weight":
                lora_module.mid_model = module
            elif lora_key == "lora_down.weight":
                lora_module.down_model = module
                lora_module.dim = weight.shape[0]
        elif lora_key in HADA_KEY:
            if type(lora_module) != LoraHadaModule:
                alpha = lora_module.alpha
                bias = lora_module.bias
                lora_module = LoraHadaModule()
                lora_module.alpha = alpha
                lora_module.bias = bias
                lora.modules[key] = lora_module
            lora_module.shape = sd_module.weight.shape
            
            weight = weight.to(device=devices.device, dtype=devices.dtype)
            weight.requires_grad_(False)
            
            if lora_key == 'hada_w1_a':
                lora_module.w1a = weight
                if lora_module.up is None:
                    lora_module.up = FakeModule(
                        lora_module.w1a,
                        lora_module.inference
                    )
            elif lora_key == 'hada_w1_b':
                lora_module.w1b = weight
                lora_module.dim = weight.shape[0]
            elif lora_key == 'hada_w2_a':
                lora_module.w2a = weight
            elif lora_key == 'hada_w2_b':
                lora_module.w2b = weight
            elif lora_key == 'hada_t1':
                lora_module.t1 = weight
                lora_module.up = FakeModule(
                    lora_module.t1,
                    lora_module.inference
                )
            elif lora_key == 'hada_t2':
                lora_module.t2 = weight
            
            if type(sd_module) == torch.nn.Linear:
                lora_module.op = torch.nn.functional.linear
            elif type(sd_module) == torch.nn.Conv2d:
                lora_module.op = torch.nn.functional.conv2d
                lora_module.extra_args = {
                    'stride': sd_module.stride,
                    'padding': sd_module.padding
                }
            else:
                assert False, f'Lora layer {key_diffusers} matched a layer with unsupported type: {type(sd_module).__name__}'
            
        else:
            assert False, f'Bad Lora layer name: {key_diffusers} - must end in lora_up.weight, lora_down.weight or alpha'

    if len(keys_failed_to_match) > 0:
        print(f"Failed to match keys when loading Lora {filename}: {keys_failed_to_match}")

    if len(keys_failed_to_match_lbw) > 0:
        print(f"Failed to match keys when loading Lora {filename}: {keys_failed_to_match_lbw}")

    return lora


def load_loras_blocks(names, lwei=None,multi=1.0):
    import lora
    loras_on_disk = [lora.available_loras.get(name, None) for name in names]
    if any([x is None for x in loras_on_disk]):
        lora.list_available_loras()

        loras_on_disk = [lora.available_loras.get(name, None) for name in names]

    for i, name in enumerate(names):
        locallora = None

        lora_on_disk = loras_on_disk[i]
        if lora_on_disk is not None:
            if locallora is None or os.path.getmtime(lora_on_disk.filename) > locallora.mtime:
                locallora = load_lora(name, lora_on_disk.filename,lwei[i])

        if locallora is None:
            print(f"Couldn't find Lora with name {name}")
            continue

        locallora.multiplier = multi
        lora.loaded_loras.append(locallora)

def smakegrid(imgs,xs,ys,currentmodel,p):
    ver_texts = [[images.GridAnnotation(y)] for y in ys]
    hor_texts = [[images.GridAnnotation(x)] for x in xs]

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(len(xs) * w, len(ys) * h), color='black')

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % len(xs) * w, i // len(xs) * h))

    grid = images.draw_grid_annotations(grid,w, h, hor_texts, ver_texts)
    grid = draw_origin(grid, currentmodel,w*len(xs),h*len(ys),w)
    if opts.grid_save:
        images.save_image(grid, opts.outdir_txt2img_grids, "xy_grid", extension=opts.grid_format, prompt=p.prompt, seed=p.seed, grid=True, p=p)

    return grid

def draw_origin(grid, text,width,height,width_one):
    grid_d= Image.new("RGB", (grid.width,grid.height), "white")
    grid_d.paste(grid,(0,0))
    def get_font(fontsize):
        try:
            return ImageFont.truetype(opts.font or Roboto, fontsize)
        except Exception:
            return ImageFont.truetype(Roboto, fontsize)
    d= ImageDraw.Draw(grid_d)
    color_active = (0, 0, 0)
    fontsize = (width+height)//25
    fnt = get_font(fontsize)

    if grid.width != width_one:
        while d.multiline_textsize(text, font=fnt)[0] > width_one*0.75 and fontsize > 0:
            fontsize -=1
            fnt = get_font(fontsize)
    d.multiline_text((0,0), text, font=fnt, fill=color_active,align="center")
    return grid_d

def newrun(p, *args):
        script_index = args[0]

        if args[0] ==0:
            script = None
            for obj in scripts.scripts_txt2img.alwayson_scripts:
                if "lora_block_weight" in obj.filename:
                    script = obj 
                    script_args = args[script.args_from:script.args_to]
        else:
            script = scripts.scripts_txt2img.selectable_scripts[script_index-1]
            
            if script is None:
                return None

            script_args = args[script.args_from:script.args_to]

        processed = script.run(p, *script_args)

        shared.total_tqdm.clear()

        return processed

def effectivechecker(imgs,ss,ls,diffcol,thresh,revxy):
    diffs = []
    outnum =[]
    imgs[0],imgs[1] = imgs[1],imgs[0]
    im1 = np.array(imgs[0])

    for i in range(len(imgs)-1):
            im2 = np.array(imgs[i+1])

            abs_diff = cv2.absdiff(im2 ,  im1)

            abs_diff_t = cv2.threshold(abs_diff, int(thresh), 255, cv2.THRESH_BINARY)[1]        
            res = abs_diff_t.astype(np.uint8)
            percentage = (np.count_nonzero(res) * 100)/ res.size
            if "white" in diffcol: abs_diff = cv2.bitwise_not(abs_diff)
            outnum.append(percentage)

            abs_diff = Image.fromarray(abs_diff)     

            diffs.append(abs_diff)
            
    outs = []
    for i in range(len(ls)):
        ls[i] = ls[i] + "\n Diff : " + str(round(outnum[i],3)) + "%"

    if not revxy:
        for diff,img in zip(diffs,imgs[1:]):
            outs.append(diff)
            outs.append(img)
            outs.append(imgs[0])
        ss = ["diff",ss[0],"source"]
        return outs,ss,ls
    else:
        outs = [imgs[0]]*len(diffs)  + imgs[1:]+ diffs
        ss = ["source",ss[0],"diff"]
        return outs,ls,ss
