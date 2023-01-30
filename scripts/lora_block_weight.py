import modules.scripts as scripts
import gradio as gr

from modules.processing import Processed
from modules.shared import opts, cmd_opts, state
from linecache import clearcache
from math import gamma
import gradio as gr
import argparse
import modules.ui
import glob
import os
import re
import torch
import math
import pprint


from modules import shared, devices, sd_models,extra_networks

from PIL import Image, PngImagePlugin
from collections import namedtuple
from modules import  sd_models,script_callbacks
from modules.ui import create_refresh_button, create_output_panel
from modules.shared import opts
from modules.sd_models import  checkpoints_loaded

LORABLOCKS=["encoder",
"down_blocks_0_attentions_0",
"down_blocks_0_attentions_1",
"down_blocks_1_attentions_0",
"down_blocks_1_attentions_1",
"down_blocks_2_attentions_0",
"down_blocks_2_attentions_1",
"mid_block_attentions_0",
"up_blocks_1_attentions_0",
"up_blocks_1_attentions_1",
"up_blocks_1_attentions_2",
"up_blocks_2_attentions_0",
"up_blocks_2_attentions_1",
"up_blocks_2_attentions_2",
"up_blocks_3_attentions_0",
"up_blocks_3_attentions_1",
"up_blocks_3_attentions_2"]



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
OUTALL:1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1\n"

        path_root = scripts.basedir()
        filepath = os.path.join(path_root,"scripts", "lbwpresets.txt")
        lbwpresets=""
        try:
            with open(filepath) as f:
                lbwpresets = f.read()
        except OSError as e:
                lbwpresets=LWEIGHTSPRESETS
  
        loraratios=lbwpresets.splitlines()
        lratios={}
        for i,l in enumerate(loraratios):
            lratios[l.split(":")[0]]=l.split(":")[1]
        rasiostags = [k for k in lratios.keys()]
        rasiostags = ",".join(rasiostags)

        with gr.Accordion("LoRA Block Weight",open = False):
            with gr.Row():
                lbw_useblocks =  gr.Checkbox(value = True,label="Active",interactive =True)
                reloadtext = gr.Button(elem_id="lora block weights", value="Reload Presets",variant='primary')
                savetext = gr.Button(elem_id="lora block weights", value="Save Presets",variant='primary')
                openeditor = gr.Button(elem_id="lora block weights", value="Open TextEditor",variant='primary')
            bw_ratiotags= gr.TextArea(label="",lines=2,value=rasiostags,visible =True,interactive =True) 
            with gr.Accordion("Weights setting",open = True):
                lbw_loraratios = gr.TextArea(label="",value=lbwpresets,visible =True,interactive  = True)      
        
        import subprocess
        def openeditors():
            subprocess.Popen(['start', filepath], shell=True)
        
        def reloadpresets():
            try:
                with open(filepath) as f:
                    return f.read()
            except OSError as e:
                pass

        def savepresets(text):
            with open(filepath,mode = 'w') as f:
                f.write(text)

        reloadtext.click(fn=reloadpresets,inputs=[],outputs=[lbw_loraratios])
        savetext.click(fn=savepresets,inputs=[lbw_loraratios],outputs=[])
        openeditor.click(fn=openeditors,inputs=[],outputs=[])

        return lbw_loraratios,lbw_useblocks

    def process(self, p, loraratios,useblocks):    
    #def process_batch(self, p, loraratios,useblocks,**args):
        if useblocks:
            loraratios=loraratios.splitlines()
            lratios={}
            for i,l in enumerate(loraratios):
                lratios[l.split(":",1)[0]]=l.split(":",1)[1]

            _, extra_network_data = extra_networks.parse_prompts(p.all_prompts[0:1])
            calledloras = extra_network_data["lora"]
            lorans = []
            lorars = []
            for called in calledloras:
                if len(called.items) <3:continue
                if called.items[2] in lratios:
                    lorans.append(called.items[0])
                    wei = lratios[called.items[2]]
                    multiple = called.items[1]
                    lorars.append([float(w) for w in wei.split(",")])
            if len(lorars) > 0: load_loras_blocks(lorans,lorars,multiple)
            
        return

    def postprocess(self, p, processed, *args):
        import lora
        lora.loaded_loras.clear()


re_digits = re.compile(r"\d+")
re_unet_down_blocks = re.compile(r"lora_unet_down_blocks_(\d+)_attentions_(\d+)_(.+)")
re_unet_mid_blocks = re.compile(r"lora_unet_mid_block_attentions_(\d+)_(.+)")
re_unet_up_blocks = re.compile(r"lora_unet_up_blocks_(\d+)_attentions_(\d+)_(.+)")
re_text_block = re.compile(r"lora_te_text_model_encoder_layers_(\d+)_(.+)")

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

    if match(m, re_text_block):
        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    return key


def load_lora(name, filename,lwei):
    import lora
    locallora = lora.LoraModule(name)
    locallora.mtime = os.path.getmtime(filename)

    sd = sd_models.read_state_dict(filename)

    keys_failed_to_match = []

    for key_diffusers, weight in sd.items():
        ratio = 1
        
        for i,block in enumerate(LORABLOCKS):
            if block in key_diffusers:
                ratio = lwei[i]
        
        weight =weight *math.sqrt(ratio)

        fullkey = convert_diffusers_name_to_compvis(key_diffusers)
        #print(key_diffusers+":"+fullkey+"x" + str(ratio))
        key, lora_key = fullkey.split(".", 1)

        sd_module = shared.sd_model.lora_layer_mapping.get(key, None)
        if sd_module is None:
            keys_failed_to_match.append(key_diffusers)
            continue

        lora_module = locallora.modules.get(key, None)
        if lora_module is None:
            lora_module = lora.LoraUpDownModule()
            locallora.modules[key] = lora_module

        if lora_key == "alpha":
            lora_module.alpha = weight.item()
            continue

        if type(sd_module) == torch.nn.Linear:
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif type(sd_module) == torch.nn.Conv2d:
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        else:
            assert False, f'Lora layer {key_diffusers} matched a layer with unsupported type: {type(sd_module).__name__}'

        with torch.no_grad():
            module.weight.copy_(weight)

        module.to(device=devices.device, dtype=devices.dtype)

        if lora_key == "lora_up.weight":
            lora_module.up = module
        elif lora_key == "lora_down.weight":
            lora_module.down = module
        else:
            assert False, f'Bad Lora layer name: {key_diffusers} - must end in lora_up.weight, lora_down.weight or alpha'

    if len(keys_failed_to_match) > 0:
        print(f"Failed to match keys when loading Lora {filename}: {keys_failed_to_match}")

    return locallora


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
