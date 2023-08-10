import cv2
import os
import gc
import re
import sys
import torch
import shutil
import math
import importlib
import numpy as np
import gradio as gr
import os.path
import random
from pprint import pprint
import modules.ui
import modules.scripts as scripts
from PIL import Image, ImageFont, ImageDraw
import modules.shared as shared
from modules import devices, sd_models, images,cmd_args, extra_networks
from modules.shared import opts, state
from modules.processing import process_images, Processed

lxyz = ""
lzyx = ""
prompts = ""
xyelem = ""
princ = False

BLOCKID26=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
BLOCKID17=["BASE","IN01","IN02","IN04","IN05","IN07","IN08","M00","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
BLOCKID12=["BASE","IN04","IN05","IN07","IN08","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05"]
BLOCKID20=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08"]
BLOCKNUMS = [12,17,20,26]
BLOCKIDS=[BLOCKID12,BLOCKID17,BLOCKID20,BLOCKID26]

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
"diffusion_model_output_blocks_11_",
"embedders"]

loopstopper = True

ATYPES =["none","Block ID","values","seed","Original Weights","elements"]

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
    def __init__(self):
        self.log = {}

    def title(self):
        return "LoRA Block Weight"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    def ui(self, is_img2img):
        LWEIGHTSPRESETS = DEF_WEIGHT_PRESET

        runorigin = scripts.scripts_txt2img.run
        runorigini = scripts.scripts_img2img.run

        scriptpath = os.path.dirname(os.path.abspath(__file__))
        path_root = scripts.basedir()
        extpath = os.path.join(scriptpath, "lbwpresets.txt")
        extpathe = os.path.join(scriptpath, "elempresets.txt")
        filepath = os.path.join(path_root,"scripts", "lbwpresets.txt")
        filepathe = os.path.join(path_root,"scripts", "elempresets.txt")

        if os.path.isfile(filepath) and not os.path.isfile(extpath):
            shutil.move(filepath,extpath)
    
        if os.path.isfile(filepathe) and not os.path.isfile(extpathe):
            shutil.move(filepathe,extpathe)

        lbwpresets=""

        try:
            with open(extpath,encoding="utf-8") as f:
                lbwpresets = f.read()
        except OSError as e:
                lbwpresets=LWEIGHTSPRESETS
                if not os.path.isfile(extpath):
                    try:
                        with open(extpath,mode = 'w',encoding="utf-8") as f:
                            f.write(lbwpresets)
                    except:
                        pass

        try:
            with open(extpathe,encoding="utf-8") as f:
                elempresets = f.read()
        except OSError as e:
                elempresets=ELEMPRESETS
                if not os.path.isfile(extpathe):
                    try:
                        with open(extpathe,mode = 'w',encoding="utf-8") as f:
                            f.write(elempresets)
                    except:
                        pass

        loraratios=lbwpresets.splitlines()
        lratios={}
        for i,l in enumerate(loraratios):
            if checkloadcond(l) : continue
            lratios[l.split(":")[0]]=l.split(":")[1]
        ratiostags = [k for k in lratios.keys()]
        ratiostags = ",".join(ratiostags)

        if os.environ.get('IGNORE_CMD_ARGS_ERRORS', None) is None:
            args = cmd_args.parser.parse_args()
        else:
            args, _ = cmd_args.parser.parse_known_args()
        if args.api:
            register()

        with gr.Accordion("LoRA Block Weight",open = False):
            with gr.Row():
                with gr.Column(min_width = 50, scale=1):
                    lbw_useblocks =  gr.Checkbox(value = True,label="Active",interactive =True,elem_id="lbw_active")
                    debug =  gr.Checkbox(value = False,label="Debug",interactive =True,elem_id="lbw_debug")
                with gr.Column(scale=5):
                    bw_ratiotags= gr.TextArea(label="",value=ratiostags,visible =True,interactive =True,elem_id="lbw_ratios") 
            with gr.Accordion("XYZ plot",open = False):
                gr.HTML(value='<p style= "word-wrap:break-word;">changeable blocks : BASE,IN00,IN01,IN02,IN03,IN04,IN05,IN06,IN07,IN08,IN09,IN10,IN11,M00,OUT00,OUT01,OUT02,OUT03,OUT04,OUT05,OUT06,OUT07,OUT08,OUT09,OUT10,OUT11</p>')
                xyzsetting = gr.Radio(label = "Active",choices = ["Disable","XYZ plot","Effective Block Analyzer"], value ="Disable",type = "index") 
                with gr.Row(visible = False) as esets:
                    diffcol = gr.Radio(label = "diff image color",choices = ["black","white"], value ="black",type = "value",interactive =True) 
                    revxy = gr.Checkbox(value = False,label="change X-Y",interactive =True,elem_id="lbw_changexy")
                    thresh = gr.Textbox(label="difference threshold",lines=1,value="20",interactive =True,elem_id="diff_thr")
                xtype = gr.Dropdown(label="X Types", choices=[x for x in ATYPES], value=ATYPES [2],interactive =True,elem_id="lbw_xtype")
                xmen = gr.Textbox(label="X Values",lines=1,value="0,0.25,0.5,0.75,1",interactive =True,elem_id="lbw_xmen")
                ytype = gr.Dropdown(label="Y Types", choices=[y for y in ATYPES], value=ATYPES [1],interactive =True,elem_id="lbw_ytype")    
                ymen = gr.Textbox(label="Y Values" ,lines=1,value="IN05-OUT05",interactive =True,elem_id="lbw_ymen")
                ztype = gr.Dropdown(label="Z type", choices=[z for z in ATYPES], value=ATYPES[0],interactive =True,elem_id="lbw_ztype")    
                zmen = gr.Textbox(label="Z values",lines=1,value="",interactive =True,elem_id="lbw_zmen")

                exmen = gr.Textbox(label="Range",lines=1,value="0.5,1",interactive =True,elem_id="lbw_exmen",visible = False) 
                eymen = gr.Textbox(label="Blocks (12ALL,17ALL,20ALL,26ALL also can be used)" ,lines=1,value="BASE,IN00,IN01,IN02,IN03,IN04,IN05,IN06,IN07,IN08,IN09,IN10,IN11,M00,OUT00,OUT01,OUT02,OUT03,OUT04,OUT05,OUT06,OUT07,OUT08,OUT09,OUT10,OUT11",interactive =True,elem_id="lbw_eymen",visible = False)  
                ecount = gr.Number(value=1, label="number of seed", interactive=True, visible = True)           

            with gr.Accordion("Weights setting",open = True):
                with gr.Row():
                    reloadtext = gr.Button(value="Reload Presets",variant='primary',elem_id="lbw_reload")
                    reloadtags = gr.Button(value="Reload Tags",variant='primary',elem_id="lbw_reload")
                    savetext = gr.Button(value="Save Presets",variant='primary',elem_id="lbw_savetext")
                    openeditor = gr.Button(value="Open TextEditor",variant='primary',elem_id="lbw_openeditor")
                lbw_loraratios = gr.TextArea(label="",value=lbwpresets,visible =True,interactive  = True,elem_id="lbw_ratiospreset")      
            
            with gr.Accordion("Elemental",open = False):  
                with gr.Row():
                    e_reloadtext = gr.Button(value="Reload Presets",variant='primary',elem_id="lbw_reload")
                    e_savetext = gr.Button(value="Save Presets",variant='primary',elem_id="lbw_savetext")
                    e_openeditor = gr.Button(value="Open TextEditor",variant='primary',elem_id="lbw_openeditor")
                elemsets = gr.Checkbox(value = False,label="print change",interactive =True,elem_id="lbw_print_change")
                elemental = gr.TextArea(label="Identifer:BlockID:Elements:Ratio,...,separated by empty line ",value = elempresets,interactive =True,elem_id="element") 

                d_true = gr.Checkbox(value = True,visible = False)
                d_false = gr.Checkbox(value = False,visible = False)
        
        import subprocess
        def openeditors(b):
            path = extpath if b else extpathe
            subprocess.Popen(['start', path], shell=True)
                  
        def reloadpresets(isweight):
            if isweight:
                try:
                    with open(extpath,encoding="utf-8") as f:
                        return f.read()
                except OSError as e:
                    pass
            else:
                try:
                    with open(extpath,encoding="utf-8") as f:
                        return f.read()
                except OSError as e:
                    pass

        def tagdicter(presets):
            presets=presets.splitlines()
            wdict={}
            for l in presets:
                if checkloadcond(l) : continue
                w=[]
                if ":" in l :
                    key = l.split(":",1)[0]
                    w = l.split(":",1)[1]
                if any(len([w for w in w.split(",")]) == x for x in BLOCKNUMS):
                    wdict[key.strip()]=w
            return ",".join(list(wdict.keys()))

        def savepresets(text,isweight):
            if isweight:
                with open(extpath,mode = 'w',encoding="utf-8") as f:
                    f.write(text)
            else:
                with open(extpathe,mode = 'w',encoding="utf-8") as f:
                    f.write(text)

        reloadtext.click(fn=reloadpresets,inputs=[d_true],outputs=[lbw_loraratios])
        reloadtags.click(fn=tagdicter,inputs=[lbw_loraratios],outputs=[bw_ratiotags])
        savetext.click(fn=savepresets,inputs=[lbw_loraratios,d_true],outputs=[])
        openeditor.click(fn=openeditors,inputs=[d_true],outputs=[])

        e_reloadtext.click(fn=reloadpresets,inputs=[d_false],outputs=[elemental])
        e_savetext.click(fn=savepresets,inputs=[elemental,d_false],outputs=[])
        e_openeditor.click(fn=openeditors,inputs=[d_false],outputs=[])

        def urawaza(active):
            if active > 0:
                register()
                scripts.scripts_txt2img.run = newrun
                scripts.scripts_img2img.run = newrun
                if active == 1:return [*[gr.update(visible = True) for x in range(6)],*[gr.update(visible = False) for x in range(4)]]
                else:return [*[gr.update(visible = False) for x in range(6)],*[gr.update(visible = True) for x in range(4)]]
            else:
                scripts.scripts_txt2img.run = runorigin
                scripts.scripts_img2img.run = runorigini
                return [*[gr.update(visible = True) for x in range(6)],*[gr.update(visible = False) for x in range(4)]]

        xyzsetting.change(fn=urawaza,inputs=[xyzsetting],outputs =[xtype,xmen,ytype,ymen,ztype,zmen,exmen,eymen,ecount,esets])

        return lbw_loraratios,lbw_useblocks,xyzsetting,xtype,xmen,ytype,ymen,ztype,zmen,exmen,eymen,ecount,diffcol,thresh,revxy,elemental,elemsets,debug

    def process(self, p, loraratios,useblocks,xyzsetting,xtype,xmen,ytype,ymen,ztype,zmen,exmen,eymen,ecount,diffcol,thresh,revxy,elemental,elemsets,debug):
        #print("self =",self,"p =",p,"presets =",loraratios,"useblocks =",useblocks,"xyzsettings =",xyzsetting,"xtype =",xtype,"xmen =",xmen,"ytype =",ytype,"ymen =",ymen,"ztype =",ztype,"zmen =",zmen)
        #Note that this does not use the default arg syntax because the default args are supposed to be at the end of the function
        if(loraratios == None):
            loraratios = DEF_WEIGHT_PRESET
        if(useblocks == None):
            useblocks = True

        lorachecker(self)
        self.log["enable LBW"] = useblocks
        self.log["registerd"] = registerd
            
        if useblocks:
            loraratios=loraratios.splitlines()
            elemental = elemental.split("\n\n") if elemental is not None else []
            lratios={}
            elementals={}
            for l in loraratios:
                if checkloadcond(l) : continue
                l0=l.split(":",1)[0]
                lratios[l0.strip()]=l.split(":",1)[1]
            for e in elemental:
                if ":" not in e: continue
                e0=e.split(":",1)[0]
                elementals[e0.strip()]=e.split(":",1)[1]
            if elemsets : print(xyelem)
            if xyzsetting and "XYZ" in p.prompt:
                lratios["XYZ"] = lxyz
                lratios["ZYX"] = lzyx
            if xyelem != "":
                if "XYZ" in elementals.keys():
                    elementals["XYZ"] = elementals["XYZ"] + ","+ xyelem
                else:
                    elementals["XYZ"] = xyelem
            self.lratios = lratios
            self.elementals = elementals
            global princ
            princ = elemsets
    
    def before_process_batch(self, p, loraratios,useblocks,xyzsetting,xtype,xmen,ytype,ymen,ztype,zmen,exmen,eymen,ecount,diffcol,thresh,revxy,elemental,elemsets,debug,**kwargs):
        if useblocks:
            if not self.isnet: p.disable_extra_networks = False
            global prompts
            prompts = kwargs["prompts"].copy()

    def process_batch(self, p, loraratios,useblocks,xyzsetting,xtype,xmen,ytype,ymen,ztype,zmen,exmen,eymen,ecount,diffcol,thresh,revxy,elemental,elemsets,debug,**kwargs):
        if useblocks:
            if not self.isnet: p.disable_extra_networks = True
            o_prompts = [p.prompt]
            for prompt in prompts:
                if "<lora" in prompt or "<lyco" in prompt:
                    o_prompts = prompts.copy()
            if not self.isnet: loradealer(self, o_prompts ,self.lratios,self.elementals)

    def postprocess(self, p, processed, presets,useblocks,xyzsetting,xtype,xmen,ytype,ymen,ztype,zmen,exmen,eymen,ecount,diffcol,thresh,revxy,elemental,elemsets,debug,*args):
        lora = importer(self)
        lora.loaded_loras.clear()
        global lxyz,lzyx,xyelem             
        lxyz = lzyx = xyelem = ""
        if debug:
            print(self.log)
        gc.collect()

    def after_extra_networks_activate(self, p, presets,useblocks,xyzsetting,xtype,xmen,ytype,ymen,ztype,zmen,exmen,eymen,ecount,diffcol,thresh,revxy,elemental,elemsets,debug, *args, **kwargs):
        if useblocks:
            loradealer(self, kwargs["prompts"] ,self.lratios,self.elementals,kwargs["extra_network_data"])

    def run(self,p,presets,useblocks,xyzsetting,xtype,xmen,ytype,ymen,ztype,zmen,exmen,eymen,ecount,diffcol,thresh,revxy,elemental,elemsets,debug):
        self.log={}
        self.log["pass XYZ"] = True
        self.log["XYZsets"] = xyzsetting
        self.log["enable LBW"] = useblocks

        if xyzsetting >0:
            lorachecker(self)
            lora = importer(self)
            loraratios=presets.splitlines()
            lratios={}
            for l in loraratios:
                if checkloadcond(l) : continue
                l0=l.split(":",1)[0]
                lratios[l0.strip()]=l.split(":",1)[1]

            if "XYZ" in p.prompt:
                base = lratios["XYZ"] if "XYZ" in lratios.keys() else "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
            else: return

            for i, all in enumerate(["12ALL","17ALL","20ALL","26ALL"]):
                if eymen == all:
                    eymen = ",".join(BLOCKIDS[i])

            if xyzsetting > 1: 
                xmen,ymen = exmen,eymen
                xtype,ytype = "values","ID"
                ebase = xmen.split(",")[1]
                ebase = [ebase.strip()]*26
                base = ",".join(ebase)
                ztype = ""
                if ecount > 1:
                    ztype = "seed"
                    zmen = ",".join([str(random.randrange(4294967294)) for x in range(int(ecount))])

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
                #print(f"weights from : {base}")
                ids = [z.strip() for z in ids.split(' ')]
                weights_t = [w.strip() for w in base.split(',')]
                blockid =  BLOCKIDS[BLOCKNUMS.index(len(weights_t))] 
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
                if "elements" in at:
                    global xyelem
                    xyelem = a

            grids = []
            images =[]

            totalcount = len(xs)*len(ys)*len(zs) if xyzsetting < 2 else len(xs)*len(ys)*len(zs)  //2 +1
            shared.total_tqdm.updateTotal(totalcount)
            xc = yc =zc = 0
            state.job_count = totalcount 
            totalcount = len(xs)*len(ys)*len(zs)
            c_base = base

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
                            p.cached_c = [None,None]
                            p.cached_uc = [None,None]
                            p.cached_hr_c = [None, None]
                            p.cached_hr_uc = [None, None]
                            processed:Processed = process_images(p)
                            images.append(processed.images[0])
                        xc += 1
                    yc += 1
                zc += 1
                origin = loranames(processed.all_prompts) + ", "+ znamer(ztype,z,base)
                images,xst,yst = effectivechecker(images,xs.copy(),ys.copy(),diffcol,thresh,revxy) if xyzsetting >1 else (images,xs.copy(),ys.copy())
                grids.append(smakegrid(images,xst,yst,origin,p))
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
    calledloras = extra_network_data["lora"] if "lyco" not in extra_network_data.keys() else extra_network_data["lyco"]
    names = ""
    for called in calledloras:
        if len(called.items) <3:continue
        names += called.items[0] 
    return names

def lorachecker(self):
    try:
        import networks
        self.isnet = True
        self.layer_name = "network_layer_name"
    except:
        self.isnet = False
        self.layer_name = "lora_layer_name"  
    try:
        import lora
        self.islora = True
    except:
        pass
    try:
        import lycoris
        self.islyco = True
    except:
        pass
    self.onlyco = (not self.islora) and self.islyco
    self.isxl = hasattr(shared.sd_model,"conditioner")
    
    self.log["isnet"] = self.isnet 
    self.log["isxl"] = self.isxl
    self.log["islora"] = self.islora

def importer(self):
    if self.onlyco:
        # lycorisモジュールを動的にインポート
        lora_module = importlib.import_module("lycoris")
        return lora_module
    else:
        # loraモジュールを動的にインポート
        lora_module = importlib.import_module("lora")
        return lora_module

def loradealer(self, prompts,lratios,elementals, extra_network_data = None):
    if extra_network_data is None:
        _, extra_network_data = extra_networks.parse_prompts(prompts)
    moduletypes = extra_network_data.keys()

    for ltype in moduletypes:
        lorans = []
        lorars = []
        multipliers = []
        elements = []
        if not (ltype == "lora" or ltype == "lyco") : continue
        for called in extra_network_data[ltype]:
            multiple = float(syntaxdealer(called.items,"unet=","te=",1))
            multipliers.append(multiple)
            if len(called.items) <3:
                continue
            lorans.append(called.items[0])
            weights = syntaxdealer(called.items,"lbw=",None,2)
            if weights in lratios or any(weights.count(",") == x - 1 for x in BLOCKNUMS):
                wei = lratios[weights] if weights in lratios else weights
                ratios = [w.strip() for w in wei.split(",")]
                for i,r in enumerate(ratios):
                    if r =="R":
                        ratios[i] = round(random.random(),3)
                    elif r == "U":
                        ratios[i] = round(random.uniform(-0.5,1.5),3)
                    elif r[0] == "X":
                        base = syntaxdealer(called.items,"x=",None, 3) if len(called.items) >= 4 else 1
                        ratios[i] = getinheritedweight(base, r)
                    else:
                        ratios[i] = float(r)
                print(f"LoRA Block weight ({ltype}): {called.items[0]}: {multiple} x {[x  for x in ratios]}")
                if len(ratios) != 26:
                    ratios = to26(ratios)
                lorars.append(ratios)
            if len(called.items) > 3:
                if syntaxdealer(called.items, "lbwe=",None,3) in elementals:
                    elements.append(elementals[called.items[3]])
                else:
                    elements.append(called.items[3])
            else:
                elements.append("")
        if len(lorars) > 0: load_loras_blocks(self, lorans,lorars,multipliers,elements,ltype)

def syntaxdealer(items,type1,type2,index): #type "unet=", "x=", "lwbe=" 
    target = [type1,type2] if type2 is not None else [type1]
    for t in target:
        for item in items:
            if t in item:
                return item.replace(t,"")
    return items[index] if "@" not in items[index] else 1

def isfloat(t):
    try:
        float(t)
        return True
    except:
        return False

re_inherited_weight = re.compile(r"X([+-])?([\d.]+)?")

def getinheritedweight(weight, offset):
    match = re_inherited_weight.search(offset)
    if match.group(1) == "+":
        return float(weight) + float(match.group(2))
    elif match.group(1) == "-":
        return float(weight) - float(match.group(2))  
    else:
        return float(weight) 

def load_loras_blocks(self, names, lwei,multipliers,elements = [],ltype = "lora"):
    oldnew=[]
    if "lora" == ltype:
        lora = importer(self)
        for l, loaded in enumerate(lora.loaded_loras):
            for n, name in enumerate(names):
                if name == loaded.name:
                    lbw(lora.loaded_loras[l],lwei[n],elements[n])
                    newname = lora.loaded_loras[l].name +"_in_LBW_"+ str(round(random.random(),3))
                    oldname = lora.loaded_loras[l].name
                    lora.loaded_loras[l].name = newname
                    oldnew.append([oldname,newname])

    elif "lyco" == ltype:
        import lycoris as lycomo
        for l, loaded in enumerate(lycomo.loaded_lycos):
            for n, name in enumerate(names):
                if name == loaded.name:
                    lbw(lycomo.loaded_lycos[l],lwei[n],elements[n])
                    lycomo.loaded_lycos[l].name = lycomo.loaded_lycos[l].name +"_in_LBW_"+ str(round(random.random(),3))
    
    try:
        import lora_ctl_network as ctl
        for old,new in oldnew:
            if old in ctl.lora_weights.keys():
                ctl.lora_weights[new] = ctl.lora_weights[old]
    except:
        pass

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

def get_font(fontsize):
    path_root = scripts.basedir()
    fontpath = os.path.join(path_root,"extensions","sd-webui-lora-block-weight","scripts", "Roboto-Regular.ttf")
    try:
        return ImageFont.truetype(opts.font or fontpath, fontsize)
    except Exception:
        return ImageFont.truetype(fontpath, fontsize)

def draw_origin(grid, text,width,height,width_one):
    grid_d= Image.new("RGB", (grid.width,grid.height), "white")
    grid_d.paste(grid,(0,0))

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

registerd = False

def register():
    global registerd
    registerd = True
    for obj in scripts.scripts_txt2img.alwayson_scripts:
        if "lora_block_weight" in obj.filename:
            if obj not in scripts.scripts_txt2img.selectable_scripts:
                scripts.scripts_txt2img.selectable_scripts.append(obj)
                scripts.scripts_txt2img.titles.append("LoRA Block Weight")
    for obj in scripts.scripts_img2img.alwayson_scripts:
        if "lora_block_weight" in obj.filename:
            if obj not in scripts.scripts_img2img.selectable_scripts:
                scripts.scripts_img2img.selectable_scripts.append(obj)
                scripts.scripts_img2img.titles.append("LoRA Block Weight")

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

def lbw(lora,lwei,elemental):
    elemental = elemental.split(",")
    for key in lora.modules.keys():
        ratio = 1
        picked = False
        errormodules = []

        for i,block in enumerate(BLOCKS):
            if block in key:
                if i == 26:
                    i = 0
                ratio = lwei[i] 
                picked = True
                currentblock = i

        if not picked:
            errormodules.append(key)
        
        if len(elemental) > 0:
            skey = key + BLOCKID26[currentblock]
            for d in elemental:
                if d.count(":") != 2 :continue
                dbs,dws,dr = (hyphener(d.split(":")[0]),d.split(":")[1],d.split(":")[2])
                dbs,dws = (dbs.split(" "), dws.split(" "))
                dbn,dbs = (True,dbs[1:]) if dbs[0] == "NOT" else (False,dbs)
                dwn,dws = (True,dws[1:]) if dws[0] == "NOT" else (False,dws)
                flag = dbn
                for db in dbs:
                    if db in skey:
                        flag = not dbn
                if flag:flag = dwn
                else:continue
                for dw in dws:
                    if dw in skey:
                        flag = not dwn
                if flag:
                    dr = float(dr)
                    if princ :print(dbs,dws,key,dr)
                    ratio = dr

        ltype = type(lora.modules[key]).__name__
        set = False
        if ltype in LORAANDSOON.keys():
            setattr(lora.modules[key],LORAANDSOON[ltype],torch.nn.Parameter(getattr(lora.modules[key],LORAANDSOON[ltype]) * ratio))
            #print(ltype)
            set = True
        else:
            if hasattr(lora.modules[key],"up_model"):
                lora.modules[key].up_model.weight= torch.nn.Parameter(lora.modules[key].up_model.weight *ratio)
                #print("LoRA using LoCON")
                set = True
            else:
                lora.modules[key].up.weight= torch.nn.Parameter(lora.modules[key].up.weight *ratio)
                #print("LoRA")
                set = True
        if not set : 
            print("unkwon LoRA")

    if len(errormodules) > 0:
        print(errormodules)
    return lora

LORAANDSOON = {
    "LoraHadaModule" : "w1a",
    "LycoHadaModule" : "w1a",
    "NetworkModuleHada": "w1a",
    "FullModule" : "weight",
    "NetworkModuleFull": "weight",
    "IA3Module" : "w",
    "NetworkModuleIa3" : "w",
    "LoraKronModule" : "w1",
    "LycoKronModule" : "w1",
    "NetworkModuleLokr": "w1",
}

def hyphener(t):
    t = t.split(" ")
    for i,e in enumerate(t):
        if "-" in e:
            e = e.split("-")
            if  BLOCKID26.index(e[1]) > BLOCKID26.index(e[0]):
                t[i] = " ".join(BLOCKID26[BLOCKID26.index(e[0]):BLOCKID26.index(e[1])+1])
            else:
                t[i] = " ".join(BLOCKID26[BLOCKID26.index(e[1]):BLOCKID26.index(e[0])+1])
    return " ".join(t)

ELEMPRESETS="\
ATTNDEEPON:IN05-OUT05:attn:1\n\n\
ATTNDEEPOFF:IN05-OUT05:attn:0\n\n\
PROJDEEPOFF:IN05-OUT05:proj:0\n\n\
XYZ:::1"

def to26(ratios):
    ids = BLOCKIDS[BLOCKNUMS.index(len(ratios))]
    output = [0]*26
    for i, id in enumerate(ids):
        output[BLOCKID26.index(id)] = ratios[i]
    return output

def checkloadcond(l:str)->bool:
    # ここの条件分岐は読み込んだ行がBlock Waightの書式にあっているかを確認している。
    # [:]が含まれ、16個(LoRa)か25個(LyCORIS),11,19(XL),のカンマが含まれる形式であるうえ、
    # それがコメントアウト行(# foobar)でないことが求められている。
    # 逆に言うとコメントアウトしたいなら絶対"# "から始めることを要求している。

    # This conditional branch is checking whether the loaded line conforms to the Block Weight format.
    # It is required that "[:]" is included, and the format contains either 16 commas (for LoRa) or 25 commas (for LyCORIS),
    # and it's not a comment line (e.g., "# foobar").
    # Conversely, if you want to comment out, it requires that it absolutely starts with "# ".
    res=(":" not in l) or (not any(l.count(",") == x - 1  for x in BLOCKNUMS)) or ("#" in l)
    #print("[debug]", res,repr(l))
    return res
