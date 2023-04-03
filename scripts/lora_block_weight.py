import cv2
import os
import gc
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
prompts = ""

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

DEF_WEIGHT_PRESET = "\
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

class Script(modules.scripts.Script):   
    def title(self):
        return "LoRA Block Weight"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    def ui(self, is_img2img):
        import lora
        LWEIGHTSPRESETS = DEF_WEIGHT_PRESET

        runorigin = scripts.scripts_txt2img.run
        runorigini = scripts.scripts_img2img.run

        path_root = scripts.basedir()
        extpath = os.path.join(path_root,"extensions","sd-webui-lora-block-weight","scripts", "lbwpresets.txt")
        filepath = os.path.join(path_root,"scripts", "lbwpresets.txt")

        if os.path.isfile(extpath) and not os.path.isfile(filepath):
            shutil.move(extpath,filepath)
            
        lbwpresets=""

        try:
            with open(filepath,encoding="utf-8") as f:
                lbwpresets = f.read()
        except OSError as e:
                lbwpresets=LWEIGHTSPRESETS
                if not os.path.isfile(filepath):
                    try:
                        with open(filepath,mode = 'w',encoding="utf-8") as f:
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
                with open(filepath,encoding="utf-8") as f:
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
            with open(filepath,mode = 'w',encoding="utf-8") as f:
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
        #Note that this does not use the default arg syntax because the default args are supposed to be at the end of the function
        if(loraratios == None):
            loraratios = DEF_WEIGHT_PRESET
        if(useblocks == None):
            useblocks = True
            
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
            self.lratios = lratios
        return
    
    def before_process_batch(self, p, loraratios,useblocks,xyzsetting,xtype,xmen,ytype,ymen,ztype,zmen,exmen,eymen,ecount,diffcol,thresh,**kwargs):
        if useblocks:
            global prompts
            prompts = kwargs["prompts"].copy()
            kwargs["prompts"], _ = extra_networks.parse_prompts(kwargs["prompts"])

    def process_batch(self, p, loraratios,useblocks,xyzsetting,xtype,xmen,ytype,ymen,ztype,zmen,exmen,eymen,ecount,diffcol,thresh,**kwargs):
        if useblocks:
            loradealer(prompts ,self.lratios)

    def postprocess(self, p, processed, *args):
        import lora
        lora.loaded_loras.clear()
        gc.collect()

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

def loradealer(prompts,lratios):
    _, extra_network_data = extra_networks.parse_prompts(prompts)
    calledloras = extra_network_data["lora"]
    lorans = []
    lorars = []
    multipliers = []
    for called in calledloras:
        lorans.append(called.items[0])
        multiple = float(called.items[1])
        multipliers.append(multiple)
        if len(called.items) <3:
            lorars.append([])
            continue
        if called.items[2] in lratios or called.items[2].count(",") ==16 or called.items[2].count(",") ==25:
            wei = lratios[called.items[2]] if called.items[2] in lratios else called.items[2] 
            ratios = [w.strip() for w in wei.split(",")]
            for i,r in enumerate(ratios):
                if r =="R":
                    ratios[i] = round(random.random(),3)
                elif r == "U":
                    ratios[i] = round(random.uniform(-0.5,1.5),3)
                elif r[0] == "X":
                    base = called.items[3] if len(called.items) >= 4 else 1
                    ratios[i] = getinheritedweight(base, r)
                else:
                    ratios[i] = float(r)
            print(f"LoRA Block weight: {called.items[0]}: {multiple} x {[x  for x in ratios]}")
            if len(ratios)==17:
                ratios = [ratios[0]] + [1] + ratios[1:3]+ [1] + ratios[3:5]+[1] + ratios[5:7]+[1,1,1] + [ratios[7]] + [1,1,1] + ratios[8:]
            lorars.append(ratios)
    if len(lorars) > 0: load_loras_blocks(lorans,lorars,multipliers)

re_inherited_weight = re.compile(r"X([+-])?([\d.]+)?")

def getinheritedweight(weight, offset):
    match = re_inherited_weight.search(offset)
    if match.group(1) == "+":
        return float(weight) + float(match.group(2))
    elif match.group(1) == "-":
        return float(weight) - float(match.group(2))  
    else:
        return float(weight) 

def load_loras_blocks(names, lwei=None,multipliers = None):
    import lora
    lora.loaded_loras.clear()

    loras_on_disk = [lora.available_loras.get(name, None) for name in names]
    if any([x is None for x in loras_on_disk]):
        lora.list_available_loras()

        loras_on_disk = [lora.available_loras.get(name, None) for name in names]

    for i, name in enumerate(names):
        lora_on_disk = loras_on_disk[i]
        locallora = lora.load_lora(name, lora_on_disk.filename)

        if locallora is None:
            print(f"Couldn't find Lora with name {name}")
            continue
        else:
            if lwei[i] != []:
                locallora = lbw(locallora,lwei[i])
        locallora.multiplier = multipliers[i]

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

def lbw(lora,lwei):
    for key in lora.modules.keys():
        ratio = 1
        picked = False
        errormodules = []

        for i,block in enumerate(BLOCKS):
            if block in key:
                ratio = lwei[i]
                picked = True
        if not picked:
            errormodules.append(key)
        
        if type(lora.modules[key]).__name__ == "LoraHadaModule":
            lora.modules[key].w1a= torch.nn.Parameter(lora.modules[key].w1a *ratio)    
        else:
            if hasattr(lora.modules[key],"up_model"):
                lora.modules[key].up_model.weight= torch.nn.Parameter(lora.modules[key].up_model.weight *ratio)
            else:
                lora.modules[key].up.weight= torch.nn.Parameter(lora.modules[key].up.weight *ratio)

    lora.name = lora.name +"added_by_lora_block_weight"+ str(random.random())
    if len(errormodules) > 0:
        print(errormodules)
    return lora
