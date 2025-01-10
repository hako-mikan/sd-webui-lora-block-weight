import cv2
import json
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
import time
from pprint import pprint
import modules.ui
import modules.scripts as scripts
from PIL import Image, ImageFont, ImageDraw
import modules.shared as shared
from modules import devices, sd_models, images,cmd_args, extra_networks, sd_hijack
from modules.shared import cmd_opts, opts, state
from modules.processing import process_images, Processed
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser

LBW_T = "customscript/lora_block_weight.py/txt2img/Active/value"
LBW_I = "customscript/lora_block_weight.py/img2img/Active/value"

if os.path.exists(cmd_opts.ui_config_file):
    with open(cmd_opts.ui_config_file, 'r', encoding="utf-8") as json_file:
        ui_config = json.load(json_file)
else:
    print("ui config file not found, using default values")
    ui_config = {}

startup_t = ui_config[LBW_T] if LBW_T in ui_config else None
startup_i = ui_config[LBW_I] if LBW_I in ui_config else None
active_t = "Active" if startup_t else "Not Active"
active_i = "Active" if startup_i else "Not Active"

lxyz = ""
lzyx = ""
prompts = ""
xyelem = ""
princ = False

try:
    from modules_forge import forge_version
    forge = True
except:
    forge = False

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
"embedders",
"transformer_resblocks"]

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

scriptpath = os.path.dirname(os.path.abspath(__file__))

class Script(modules.scripts.Script):
    def __init__(self):
        self.log = {}
        self.stops = {}
        self.starts = {}
        self.active = False
        self.lora = {}
        self.lycoris = {}
        self.networks = {}

        self.stopsf = []
        self.startsf = []
        self.uf = []
        self.lf = []
        self.ef = []

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

        with gr.Accordion(f"LoRA Block Weight : {active_i if is_img2img else active_t}",open = False) as acc:
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
            
            with gr.Accordion("Make Weights",open = False):  
                with gr.Row():
                    m_text = gr.Textbox(value="",label="Weights")
                with gr.Row():
                    m_add = gr.Button(value="Add to presets",size="sm",variant='primary')
                    m_add_save = gr.Button(value="Add to presets and Save",size="sm",variant='primary')
                    m_name = gr.Textbox(value="",label="Identifier")
                with gr.Row():
                    m_type = gr.Radio(label="Weights type",choices=["17(1.X/2.X)", "26(1.X/2.X full)", "12(XL)","20(XL full)"], value="17(1.X/2.X)")
                with gr.Row():
                    m_set_0 = gr.Button(value="Set All 0",variant='primary')
                    m_set_1 = gr.Button(value="Set All 1",variant='primary')
                    m_custom = gr.Button(value="Set custom",variant='primary')
                    m_custom_v = gr.Slider(show_label=False, minimum=-1.0, maximum=1, step=0.1, value=0, interactive=True)
                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                            gr.Slider(visible=False)
                    with gr.Column(scale=2, min_width=200):
                        base = gr.Slider(label="BASE", minimum=-1, maximum=1, step=0.1, value=0.0)
                    with gr.Column(scale=1, min_width=100):
                        gr.Slider(visible=False)
                with gr.Row():
                    with gr.Column(scale=2, min_width=200):
                        ins = [gr.Slider(label=block, minimum=-1.0, maximum=1, step=0.1, value=0, interactive=True) for block in BLOCKID26[1:13]]
                    with gr.Column(scale=2, min_width=200):
                        outs = [gr.Slider(label=block, minimum=-1.0, maximum=1, step=0.1, value=0, interactive=True) for block in reversed(BLOCKID26[14:])]
                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                        gr.Slider(visible=False)
                    with gr.Column(scale=2, min_width=200):
                        m00 = gr.Slider(label="M00", minimum=-1, maximum=1, step=0.1, value=0.0)
                    with gr.Column(scale=1, min_width=100):
                        gr.Slider(visible=False)

                    blocks = [base] + ins + [m00] + outs[::-1]
                    for block in blocks:
                        if block.label not in BLOCKID17:
                            block.visible = False

                m_set_0.click(fn=lambda x:[0]*26 + [",".join(["0"]*int(x[:2]))],inputs=[m_type],outputs=blocks + [m_text])
                m_set_1.click(fn=lambda x:[1]*26 + [",".join(["1"]*int(x[:2]))],inputs=[m_type],outputs=blocks + [m_text])
                m_custom.click(fn=lambda x,y:[x]*26 + [",".join([str(x)]*int(y[:2]))],inputs=[m_custom_v,m_type],outputs=blocks + [m_text])

                def addweights(weights, id, presets, save = False):
                    if id == "":id = "NONAME"
                    lines = presets.strip().split("\n")
                    id_found = False
                    for i, line in enumerate(lines):
                        if line.startswith("#"):
                            continue
                        if line.split(":")[0] == id:
                            lines[i] = f"{id}:{weights}"
                            id_found = True
                            break
                    if not id_found:
                        lines.append(f"{id}:{weights}")

                    if save:
                        with open(extpath,mode = 'w',encoding="utf-8") as f:
                            f.write("\n".join(lines))

                    return "\n".join(lines)

                def changetheblocks(sdver,*blocks):
                    sdver = int(sdver[:2])
                    output = []
                    targ_blocks = BLOCKIDS[BLOCKNUMS.index(sdver)]
                    for i, block in enumerate(BLOCKID26):
                        if block in targ_blocks:
                            output.append(str(blocks[i]))
                    return [",".join(output)] + [gr.update(visible = True if block in targ_blocks else False) for block in BLOCKID26]
                
                m_add.click(fn=addweights, inputs=[m_text,m_name,lbw_loraratios],outputs=[lbw_loraratios])
                m_add_save.click(fn=addweights, inputs=[m_text,m_name,lbw_loraratios, d_true],outputs=[lbw_loraratios])
                m_type.change(fn=changetheblocks, inputs=[m_type] + blocks,outputs=[m_text] + blocks)

                d_true = gr.Checkbox(value = True,visible = False)
                d_false = gr.Checkbox(value = False,visible = False)

            lbw_useblocks.change(fn=lambda x:gr.update(label = f"LoRA Block Weight : {'Active' if x else 'Not Active'}"),inputs=lbw_useblocks, outputs=[acc])

        def makeweights(sdver, *blocks):
            sdver = int(sdver[:2])
            output = []
            targ_blocks = BLOCKIDS[BLOCKNUMS.index(sdver)]
            for i, block in enumerate(BLOCKID26):
                if block in targ_blocks:
                    output.append(str(blocks[i]))
            return ",".join(output)

        changes = [b.release(fn=makeweights,inputs=[m_type] + blocks,outputs=[m_text]) for b in blocks]

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
            self.active = True
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

            if not hasattr(self,"lbt_dr_callbacks"):
                self.lbt_dr_callbacks = on_cfg_denoiser(self.denoiser_callback)

    def denoiser_callback(self, params: CFGDenoiserParams):
        def setparams(self, key, te, u ,sets):
            for dicts in [self.lora,self.lycoris,self.networks]:
                for lora in dicts:
                    if lora.name.split("_in_LBW_")[0] == key:
                        lora.te_multiplier = te
                        lora.unet_multiplier = u
                        sets.append(key)

        if forge and self.active:
            def apply_weight(stop = False):
                if not stop:
                    flag_step = self.startsf
                else:
                    flag_step = self.stopsf

                lora_patches = shared.sd_model.forge_objects.unet.lora_patches
                refresh_keys = {}
                for m, l, e, s, (patch_key, lora_patch) in zip(self.uf, self.lf, self.ef, flag_step, list(lora_patches.items())):
                    refresh = False
                    for key, vals in lora_patch.items():
                        n_vals = []
                        for v in [v for v in vals if v[1][0] in LORAS]:
                            if s is not None and s == params.sampling_step:
                                if not stop:
                                    ratio, _ = ratiodealer(key.replace(".","_"), l, e)
                                    n_vals.append((ratio * m, *v[1:]))
                                else:
                                    n_vals.append((0, *v[1:]))
                                refresh = True
                            else:
                                n_vals.append(v)
                        lora_patch[key] = n_vals
                    if refresh:
                        refresh_keys[patch_key] = None

                if len(refresh_keys):
                    for refresh_key in list(refresh_keys.keys()):
                        patch = lora_patches[refresh_key]
                        del lora_patches[refresh_key]
                        new_key = (f"{refresh_key[0]}_{str(time.time())}", *refresh_key[1:])
                        refresh_keys[refresh_key] = new_key
                        lora_patches[new_key] = patch

                    shared.sd_model.forge_objects.unet.refresh_loras()

                    for refresh_key, new_key in list(refresh_keys.items()):
                        patch = lora_patches[new_key]
                        del lora_patches[new_key]
                        lora_patches[refresh_key] = patch

            if params.sampling_step in self.startsf:
                apply_weight()

            if params.sampling_step in self.stopsf:
                apply_weight(stop=True)

        elif self.active:
            if self.starts and params.sampling_step == 0:
                for key, step_te_u in self.starts.items():
                    setparams(self, key, 0, 0, [])
                    #print("\nstart 0", self, key, 0, 0, [])

            if self.starts:
                sets = []
                for key, step_te_u in self.starts.items():
                    step, te, u = step_te_u
                    if params.sampling_step > step - 2:
                        setparams(self, key, te, u, sets)
                        #print("\nstart", self, key, u, te, sets)
                for key in sets:
                    if key in self.starts:
                        del self.starts[key]

            if self.stops:
                sets = []
                for key, step in self.stops.items():
                    if params.sampling_step > step - 2:
                        setparams(self, key, 0, 0, sets)
                        #print("\nstop", self, key, 0, 0, sets)
                for key in sets:
                    if key in self.stops:
                        del self.stops[key]
    
    def before_process_batch(self, p, loraratios,useblocks,*args,**kwargs):
        if useblocks:
            resetmemory()
            if not self.isnet: p.disable_extra_networks = False
            global prompts
            prompts = kwargs["prompts"].copy()

        if forge:
            sd_models.model_data.get_sd_model().current_lora_hash = None
            shared.sd_model.forge_objects_after_applying_lora.unet.forge_unpatch_model()
            shared.sd_model.forge_objects_after_applying_lora.clip.patcher.forge_unpatch_model()

    def process_batch(self, p, loraratios,useblocks,*args,**kwargs):
        if useblocks:
            if not self.isnet: p.disable_extra_networks = True

            o_prompts = [p.prompt]
            for prompt in prompts:
                if "<lora" in prompt or "<lyco" in prompt:
                    o_prompts = prompts.copy()
            if not self.isnet: loradealer(self, o_prompts ,self.lratios,self.elementals)

    def postprocess(self, p, processed, presets,useblocks,xyzsetting,xtype,xmen,ytype,ymen,ztype,zmen,exmen,eymen,ecount,diffcol,thresh,revxy,elemental,elemsets,debug,*args):
        if not useblocks:
            return
        lora = importer(self)
        emb_db = modules.textual_inversion.textual_inversion.EmbeddingDatabase()

        for net in lora.loaded_loras:
            if hasattr(net,"bundle_embeddings"):
                for embedding in net.bundle_embeddings.values():
                    if embedding.loaded:
                        emb_db.register_embedding(embedding)

        lora.loaded_loras.clear()

        if forge:
            sd_models.model_data.get_sd_model().current_lora_hash = None
            shared.sd_model.forge_objects_after_applying_lora.unet.forge_unpatch_model()
            shared.sd_model.forge_objects_after_applying_lora.clip.patcher.forge_unpatch_model()

        global lxyz,lzyx,xyelem             
        lxyz = lzyx = xyelem = ""
        if debug:
            print(self.log)
        gc.collect()

    def after_extra_networks_activate(self, p, presets,useblocks, *args, **kwargs):
        if useblocks:
            loradealer(self, kwargs["prompts"] ,self.lratios,self.elementals,kwargs["extra_network_data"])

    def run(self,p,presets,useblocks,xyzsetting,xtype,xmen,ytype,ymen,ztype,zmen,exmen,eymen,ecount,diffcol,thresh,revxy,elemental,elemsets,debug):
        if not useblocks:
            return
        self.__init__()
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

            generatedbases=[]
            def xyzdealer(a,at):
                nonlocal ids,alpha,p,base,c_base,generatedbases
                if "ID" in at:return
                if "values" in at:alpha = a
                if "seed" in at:
                    p.seed = int(a)
                    generatedbases=[]
                if "Weights" in at:base =c_base = lratios[a]
                if "elements" in at:
                    global xyelem
                    xyelem = a

            def imagedupewatcher(baselist,basetocheck,currentiteration):
                for idx,alreadygenerated in enumerate(baselist):
                    if (basetocheck == alreadygenerated):
                        # E.g., we already generated IND+OUTS and this is now OUTS+IND with identical weights.
                        baselist.insert(currentiteration-1, basetocheck)
                        return idx
                return -1

            def strThree(someNumber): # Returns 1.12345 as 1.123 and 1.0000 as 1
                return format(someNumber, ".3f").rstrip('0').rstrip('.')

            # Adds X and Y together using array addition.
            # If both X and Y have a value in the same block then Y's is set to 0;
            # both values are used due to both XY and YX being generated, but the diagonal then only show the first value.
            # imagedupwatcher prevents duplicate images from being generated;
            # when X and Y have non-overlapping blocks then the upper triangular images are identical to the lower ones.
            def xyoriginalweightsdealer(x,y):
                xweights = np.asarray(lratios[x].split(','), dtype=np.float32) # np array easier to add later
                yweights = np.asarray(lratios[y].split(','), dtype=np.float32)
                for idx,xval in np.ndenumerate(xweights):
                    yval = yweights[idx]
                    if xval != 0 and yval != 0:
                        yweights[idx] = 0
                # Add xweights to yweights, round to 3 places,
                # map floats to string with format of 3 decimals trailing zeroes and decimal stripped
                baseListToStrings = list(map(strThree, np.around(np.add(xweights,yweights,),3).tolist()))
                return ",".join(baseListToStrings)

            grids = []
            images =[]

            totalcount = len(xs)*len(ys)*len(zs) if xyzsetting < 2 else len(xs)*len(ys)*len(zs)  //2 +1
            shared.total_tqdm.updateTotal(totalcount)
            xc = yc =zc = 0
            state.job_count = totalcount 
            totalcount = len(xs)*len(ys)*len(zs)
            c_base = base

            for z in zs:
                generatedbases=[] 
                images = []
                yc = 0
                xyzdealer(z,ztype)
                for y in ys:
                    xc = 0
                    xyzdealer(y,ytype)
                    for x in xs:
                        xyzdealer(x,xtype)
                        if "Weights" in xtype and "Weights" in ytype:
                            c_base = xyoriginalweightsdealer(x,y)
                        else:
                            if "ID" in xtype:
                                if "values" in ytype:c_base = weightsdealer(y,x,base)
                                if "values" in ztype:c_base = weightsdealer(z,x,base)
                            if "ID" in ytype:
                                if "values" in xtype:c_base = weightsdealer(x,y,base)
                                if "values" in ztype:c_base = weightsdealer(z,y,base)
                        if "ID" in ztype:
                            if "values" in xtype:c_base = weightsdealer(x,z,base)
                            if "values" in ytype:c_base = weightsdealer(y,z,base)

                        iteration = len(xs)*len(ys)*zc + yc*len(xs) +xc +1
                        print(f"X:{xtype}, {x},Y: {ytype},{y}, Z:{ztype},{z}, base:{c_base} ({iteration}/{totalcount})")

                        dupe_index = imagedupewatcher(generatedbases,c_base,iteration)
                        if dupe_index > -1:
                            print(f"Skipping generation of duplicate base:{c_base}")
                            images.append(images[dupe_index].copy())
                            xc += 1
                            continue

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
                            generatedbases.insert(iteration-1, c_base)
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

def resetmemory():
    try:
        import networks as nets
        nets.networks_in_memory = {}
        gc.collect()

    except:
        pass

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
        te_multipliers = []
        unet_multipliers = []
        elements = []
        starts = []
        stops = []
        fparams = []
        load = False
        go_lbw = False
        
        if not (ltype == "lora" or ltype == "lyco") : continue
        for called in extra_network_data[ltype]:
            items = called.items
            setnow = False
            name = items[0]
            te = syntaxdealer(items,"te=",1)
            unet = syntaxdealer(items,"unet=",2)
            te,unet = multidealer(te,unet)

            weights = syntaxdealer(items,"lbw=",2) if syntaxdealer(items,"lbw=",2) is not None else syntaxdealer(items,"w=",2)
            elem = syntaxdealer(items, "lbwe=",3)
            start = syntaxdealer(items,"start=",None)
            stop = syntaxdealer(items,"stop=",None)
            start, stop = stepsdealer(syntaxdealer(items,"step=",None), start, stop)
            
            if weights is not None and (weights in lratios or any(weights.count(",") == x - 1 for x in BLOCKNUMS)):
                wei = lratios[weights] if weights in lratios else weights
                ratios = [w.strip() for w in wei.split(",")]
                for i,r in enumerate(ratios):
                    if r =="R":
                        ratios[i] = round(random.random(),3)
                    elif r == "U":
                        ratios[i] = round(random.uniform(-0.5,1.5),3)
                    elif r[0] == "X":
                        base = syntaxdealer(items,"x=", 3) if len(items) >= 4 else 1
                        ratios[i] = getinheritedweight(base, r)
                    else:
                        ratios[i] = float(r)
                        
                if len(ratios) != 26:
                    ratios = to26(ratios)
                setnow = True
            else:
                ratios = [1] * 26

            if elem in elementals:
                setnow = True
                elem = elementals[elem]
            else:
                elem = ""

            if setnow:
                print(f"LoRA Block weight ({ltype}): {name}: (Te:{te},Unet:{unet}) x {ratios}")
                go_lbw = True
            fparams.append([unet,ratios,elem])
            settolist([lorans,te_multipliers,unet_multipliers,lorars,elements,starts,stops],[name,te,unet,ratios,elem,start,stop])
            self.log[name] = [te,unet,ratios,elem,start,stop]

            if start:
                self.starts[name] = [int(start),te,unet]
                self.log["starts"] = load = True

            if stop:
                self.stops[name] = int(stop)
                self.log["stops"] = load = True
        
        self.startsf = [int(s) if s is not None else None for s in starts]
        self.stopsf = [int(s) if s is not None else None for s in stops]
        self.uf = unet_multipliers
        self.lf = lorars
        self.ef = elements

        if self.isnet: ltype = "nets"
        if forge: ltype = "forge"
        if go_lbw or load: load_loras_blocks(self, lorans,lorars,te_multipliers,unet_multipliers,elements,ltype, starts=starts)

def stepsdealer(step, start, stop):
    if step is None or "-" not in step:
        return start, stop
    return step.split("-")

def settolist(ls,vs):
    for l, v in zip(ls,vs):
        l.append(v)

def syntaxdealer(items,target,index): #type "unet=", "x=", "lwbe=" 
    for item in items:
        if target in item:
            return item.replace(target,"")
    if index is None or index + 1> len(items): return None
    if "=" in items[index]:return None
    return items[index] if "@" not in items[index] else 1

def isfloat(t):
    try:
        float(t)
        return True
    except:
        return False

def multidealer(t, u):
    if t is None and u is None:
        return 1,1
    elif t is None:
        return float(u),float(u)
    elif u is None:
        return float(t), float(t)
    else:
        return float(t),float(u)

re_inherited_weight = re.compile(r"X([+-])?([\d.]+)?")

def getinheritedweight(weight, offset):
    match = re_inherited_weight.search(offset)
    if match.group(1) == "+":
        return float(weight) + float(match.group(2))
    elif match.group(1) == "-":
        return float(weight) - float(match.group(2))  
    else:
        return float(weight) 

def load_loras_blocks(self, names, lwei,te,unet,elements,ltype = "lora", starts = None):
    oldnew=[]
    if "lora" == ltype:
        lora = importer(self)
        self.lora = lora.loaded_loras
        for loaded in lora.loaded_loras:
            for n, name in enumerate(names):
                if name == loaded.name:
                    if lwei[n] == [1] * 26 and elements[n] == "": continue
                    lbw(loaded,lwei[n],elements[n])
                    setall(loaded,te[n],unet[n])
                    newname = loaded.name +"_in_LBW_"+ str(round(random.random(),3))
                    oldname = loaded.name
                    loaded.name = newname
                    oldnew.append([oldname,newname])

    elif "lyco" == ltype:
        import lycoris as lycomo
        self.lycoris = lycomo.loaded_lycos
        for loaded in lycomo.loaded_lycos:
            for n, name in enumerate(names):
                if name == loaded.name:
                    lbw(loaded,lwei[n],elements[n])
                    setall(loaded,te[n],unet[n])

    elif "nets" == ltype:
        import networks as nets
        self.networks = nets.loaded_networks
        for loaded in nets.loaded_networks:
            for n, name in enumerate(names):
                if name == loaded.name:
                    lbw(loaded,lwei[n],elements[n])
                    setall(loaded,te[n],unet[n])
    
    elif "forge" == ltype:
        lora_patches = shared.sd_model.forge_objects_after_applying_lora.unet.lora_patches
        lbwf(lora_patches, unet, lwei, elements, starts)

        lora_patches = shared.sd_model.forge_objects_after_applying_lora.clip.patcher.lora_patches
        lbwf(lora_patches, te, lwei, elements, starts)

    try:
        import lora_ctl_network as ctl
        for old,new in oldnew:
            if old in ctl.lora_weights.keys():
                ctl.lora_weights[new] = ctl.lora_weights[old]
    except:
        pass

def setall(m,te,unet):
    m.name = m.name + "_in_LBW_"+ str(round(random.random(),3))
    m.te_multiplier = te
    m.unet_multiplier = unet
    m.multiplier = unet

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
    fontpath = os.path.join(scriptpath, "Roboto-Regular.ttf")
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
    txt2img = isinstance(p, modules.processing.StableDiffusionProcessingTxt2Img)

    script_index = args[0]

    if args[0] ==0:
        script = None
        for obj in scripts.scripts_txt2img.alwayson_scripts if txt2img else scripts.scripts_img2img.alwayson_scripts:
            if "lora_block_weight" in obj.filename:
                script = obj 
                script_args = args[script.args_from:script.args_to]
    else:
        script = scripts.scripts_txt2img.selectable_scripts[script_index-1] if txt2img else scripts.scripts_img2img.selectable_scripts[script_index-1]
        
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
    orig = imgs[1]
    imgs = imgs[::2]
    diffs = []
    outnum =[]

    for img in imgs:
        abs_diff = cv2.absdiff(np.array(img) ,  np.array(orig))

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
        for diff,img in zip(diffs,imgs):
            outs.append(diff)
            outs.append(img)
            outs.append(orig)
        ss = ["diff",ss[0],"source"]
        return outs,ss,ls
    else:
        outs = [orig]*len(diffs)  + imgs + diffs
        ss = ["source",ss[0],"diff"]
        return outs,ls,ss

def lbw(lora,lwei,elemental):
    errormodules = []
    for key in lora.modules.keys():
        ratio, errormodule = ratiodealer(key, lwei, elemental)
        if errormodule:
            errormodules.append(errormodule)

        ltype = type(lora.modules[key]).__name__
        set = False
        if ltype in LORAANDSOON.keys():
            if "OFT" not in ltype:
                setattr(lora.modules[key],LORAANDSOON[ltype],torch.nn.Parameter(getattr(lora.modules[key],LORAANDSOON[ltype]) * ratio))
            else:
                setattr(lora.modules[key],LORAANDSOON[ltype],getattr(lora.modules[key],LORAANDSOON[ltype]) * ratio)
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

LORAS = ["lora", "loha", "lokr"]

def lbwf(after_applying_lora_patches, ms, lwei, elements, starts):
    errormodules = []
    dict_lora_patches = dict(after_applying_lora_patches.items())

    for m, l, e, s, hash in zip(ms, lwei, elements, starts, list(after_applying_lora_patches.keys())):
        lora_patches = None
        for k, v in dict_lora_patches.items():
            if k[0] == hash[0]:
                hash = k
                lora_patches = v
                del dict_lora_patches[k]
                break
        if lora_patches is None:
            continue
        for key, vals in lora_patches.items():
            n_vals = []
            lvs = [v for v in vals if v[1][0] in LORAS]
            for v in lvs:
                ratio, errormodule = ratiodealer(key.replace(".","_"), l, e)
                n_vals.append((ratio * m if s is None or s == 0 else 0, *v[1:]))
                if errormodule:
                    errormodules.append(errormodule)
            lora_patches[key] = n_vals
            # print("[DEBUG]", hash[0], *[n_val[0] for n_val in n_vals])

        lbw_key = ",".join([str(m)] + [str(int(w) if type(w) is int or w.is_integer() else float(w)) for w in l])
        new_hash = (hash[0], lbw_key, *hash[2:])

        after_applying_lora_patches[new_hash] = after_applying_lora_patches[hash]
        if new_hash != hash:
            del after_applying_lora_patches[hash]

    if len(errormodules) > 0:
        print("Unknown modules:", errormodules)

def ratiodealer(key, lwei, elemental):
    ratio = 1
    picked = False
    errormodules = []
    currentblock = 0
    elemental = elemental.split(",")
    
    for i,block in enumerate(BLOCKS):
        if block in key:
            if i == 26 or i == 27:
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
    
    return ratio, errormodules

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
    "NetworkModuleGLora": "w1a",
    "NetworkModuleNorm": "w_norm",
    "NetworkModuleOFT": "scale"
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
