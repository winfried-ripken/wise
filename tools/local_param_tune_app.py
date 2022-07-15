import os
import random
import string
import sys

PACKAGE_PARENT = '../'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from effects.gauss2d_xy_separated import Gauss2DEffect
import helpers.session_state as session_state

from pathlib import Path
import torch.nn.functional as F
import torch

import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Specify canvas parameters in application
from helpers import make_save_path, torch_to_np, np_to_torch, save_as_image
from effects import get_default_settings

st.set_page_config(layout="wide")
state = session_state.get(file_load_key="123", canvas_key="canvas")

device = st.sidebar.selectbox(
    "Device:", ("cpu", "cuda:0")
)
im_path = st.sidebar.file_uploader("Load content image:", type=["png", "jpg"])
if not im_path:
    st.info("please select an image first")
    st.stop()

img_org = Image.open(im_path)

drawing_mode = st.sidebar.selectbox(
    "Effect:", ("xdog", "toon", "watercolor")
)

effect, preset, param_set = get_default_settings(drawing_mode)
effect.enable_checkpoints()
effect.to(device)
org_cuda = np_to_torch(img_org).to(device)

vp_path = st.sidebar.file_uploader("Load visual parameters:", type=["pt"], key=state.file_load_key)
if vp_path:
    vp = torch.load(vp_path).detach().clone()
    vp = F.interpolate(vp, (img_org.height, img_org.width))

    for i in range(vp.size(1)):
        Path("vps/").mkdir(exist_ok=True)
        save_as_image(vp[:, i], f"vps/{i}.png")

    torch.save(vp, "vp_buffer.pt")
    state.file_load_key = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
elif Path("vp_buffer.pt").exists():
    vp = torch.load("vp_buffer.pt")
else:
    vp = effect.vpd.preset_tensor(preset, org_cuda, add_local_dims=True)
    torch.save(vp, "vp_buffer.pt")

active_param = st.sidebar.selectbox("active parameter: ", ["smooth"] + param_set)

st.sidebar.text("Drawing options")
if active_param != "smooth":
    overlay = st.sidebar.slider("show parameter overlay: ", 0.0, 1.0, 0.4, 0.02)
    plus_or_minus = st.sidebar.slider("Increase or decrease param map: ", -1.0, 1.0, 0.0, 0.05)
else:
    sigma = st.sidebar.slider("Sigma: ", 0.1, 10.0, 0.5, 0.1)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 50, 3)
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)

st.sidebar.text("Update:")
realtime_update = st.sidebar.checkbox("Update in realtime", True)
invert_selection = st.sidebar.checkbox("Invert Selection", False)

# img = (np.random.random((256, 256, 4)) * 255.0).astype(np.uint8)
# img[:, :, 3] = 25
# img = Image.fromarray(img)

basewidth = 670
wpercent = (basewidth / float(img_org.size[0]))
hsize = int((float(img_org.size[1]) * float(wpercent)))
img = np.array(img_org.resize((basewidth, hsize), Image.ANTIALIAS)) / (255.0 * 2.0) + 0.5
img[:, :, 0] = 0

if active_param != "smooth":
    img[:, :, 0] += (F.interpolate(vp, (hsize, basewidth))[:,
                     effect.vpd.name2idx[active_param]].squeeze().detach().cpu().numpy() + 0.5) * overlay

img = Image.fromarray((img * 255.0).astype(np.uint8))

coll1, coll2 = st.columns(2)
coll1.header("Draw Mask")
coll2.header("Live Result")

with coll1:
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        background_image=img,
        update_streamlit=realtime_update,
        width=img.width,
        height=img.height,
        drawing_mode=drawing_mode,
        key=state.canvas_key,
    )

res_data = None
if canvas_result.image_data is not None:
    abc = np_to_torch(canvas_result.image_data.astype(np.float)).sum(dim=1, keepdim=True).to(device)

    if invert_selection:
        abc = abc * (- 1.0) + 1.0

    res_data = F.interpolate(abc, (img_org.height, img_org.width)).squeeze(1)

    if active_param != "smooth":
        vp[:, effect.vpd.name2idx[active_param]] += plus_or_minus * res_data
        vp.clamp_(-0.5, 0.5)
    else:
        gauss2dx = Gauss2DEffect(dxdy=[1.0, 0.0], dim_kernsize=5)
        gauss2dy = Gauss2DEffect(dxdy=[0.0, 1.0], dim_kernsize=5)

        vp_smoothed = gauss2dx(vp, torch.tensor(sigma).to(device))
        vp_smoothed = gauss2dy(vp_smoothed, torch.tensor(sigma).to(device))

        print(res_data.shape)
        print(vp.shape)
        print(vp_smoothed.shape)
        vp = torch.lerp(vp, vp_smoothed, res_data.unsqueeze(1))

if len(effect.vpd.name2idx) != vp.size(1):
    raise ValueError("You have loaded parameters for a different effect. Please select a different parameter file.")

with torch.no_grad():
    result_cuda = effect(org_cuda, vp)

img_res = Image.fromarray((torch_to_np(result_cuda) * 255.0).astype(np.uint8))
coll2.image(img_res.resize((basewidth, hsize), Image.ANTIALIAS))

ppp = st.columns(len(param_set))
for i, p in enumerate(param_set):
    idx = effect.vpd.name2idx[p]
    iii = F.interpolate(vp[:, idx:idx + 1] + 0.5, (int(hsize * 0.2), int(basewidth * 0.2)))

    ppp[i].text(p[:7])
    ppp[i].image(torch_to_np(iii), clamp=True)

btn = st.sidebar.button("Write changes")

if btn:
    torch.save(vp, "vp_buffer.pt")
    ttt = torch.clamp(result_cuda - 0.4 * res_data, 0.0, 1.0)

    out_path = make_save_path(str(Path(f"{im_path.name}_results/step.png")))
    img_res.save(out_path)
    torch.save(vp, f"{out_path}_vp.pt")

    if res_data is not None:
        img_resdata = Image.fromarray((torch_to_np(ttt) * 255.0).astype(np.uint8))
        img_resdata.save(f"{out_path}_mask.png")

    st.info(f"changes written to {out_path}")
