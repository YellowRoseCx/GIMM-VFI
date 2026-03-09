import re

with open("src/models/generalizable_INR/modules/softsplat.py", "r") as f:
    code = f.read()

reps = [
    ("intNorthwestY_int \* stride_out_h \+ intNorthwestX_int \* stride_out_w",
     "tl.where(intNorthwestY_int >= 0, tl.where(intNorthwestY_int < H, intNorthwestY_int, H - 1), 0) * stride_out_h + tl.where(intNorthwestX_int >= 0, tl.where(intNorthwestX_int < W, intNorthwestX_int, W - 1), 0) * stride_out_w"),
    ("intNortheastY_int \* stride_out_h \+ intNortheastX_int \* stride_out_w",
     "tl.where(intNortheastY_int >= 0, tl.where(intNortheastY_int < H, intNortheastY_int, H - 1), 0) * stride_out_h + tl.where(intNortheastX_int >= 0, tl.where(intNortheastX_int < W, intNortheastX_int, W - 1), 0) * stride_out_w"),
    ("intSouthwestY_int \* stride_out_h \+ intSouthwestX_int \* stride_out_w",
     "tl.where(intSouthwestY_int >= 0, tl.where(intSouthwestY_int < H, intSouthwestY_int, H - 1), 0) * stride_out_h + tl.where(intSouthwestX_int >= 0, tl.where(intSouthwestX_int < W, intSouthwestX_int, W - 1), 0) * stride_out_w"),
    ("intSoutheastY_int \* stride_out_h \+ intSoutheastX_int \* stride_out_w",
     "tl.where(intSoutheastY_int >= 0, tl.where(intSoutheastY_int < H, intSoutheastY_int, H - 1), 0) * stride_out_h + tl.where(intSoutheastX_int >= 0, tl.where(intSoutheastX_int < W, intSoutheastX_int, W - 1), 0) * stride_out_w"),

    ("intNorthwestY_int \* stride_outg_h \+ intNorthwestX_int \* stride_outg_w",
     "tl.where(intNorthwestY_int >= 0, tl.where(intNorthwestY_int < H, intNorthwestY_int, H - 1), 0) * stride_outg_h + tl.where(intNorthwestX_int >= 0, tl.where(intNorthwestX_int < W, intNorthwestX_int, W - 1), 0) * stride_outg_w"),
    ("intNortheastY_int \* stride_outg_h \+ intNortheastX_int \* stride_outg_w",
     "tl.where(intNortheastY_int >= 0, tl.where(intNortheastY_int < H, intNortheastY_int, H - 1), 0) * stride_outg_h + tl.where(intNortheastX_int >= 0, tl.where(intNortheastX_int < W, intNortheastX_int, W - 1), 0) * stride_outg_w"),
    ("intSouthwestY_int \* stride_outg_h \+ intSouthwestX_int \* stride_outg_w",
     "tl.where(intSouthwestY_int >= 0, tl.where(intSouthwestY_int < H, intSouthwestY_int, H - 1), 0) * stride_outg_h + tl.where(intSouthwestX_int >= 0, tl.where(intSouthwestX_int < W, intSouthwestX_int, W - 1), 0) * stride_outg_w"),
    ("intSoutheastY_int \* stride_outg_h \+ intSoutheastX_int \* stride_outg_w",
     "tl.where(intSoutheastY_int >= 0, tl.where(intSoutheastY_int < H, intSoutheastY_int, H - 1), 0) * stride_outg_h + tl.where(intSoutheastX_int >= 0, tl.where(intSoutheastX_int < W, intSoutheastX_int, W - 1), 0) * stride_outg_w"),
]

for old, new in reps:
    code = re.sub(old, new, code)

with open("src/models/generalizable_INR/modules/softsplat.py", "w") as f:
    f.write(code)
