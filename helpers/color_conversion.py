import torch
from helpers.index_helper import IndexHelper


def hue_to_rgb(h):
    i = IndexHelper(h)
    r = torch.abs(h * 6.0 - 3.0) - 1.0
    g = 2.0 - torch.abs(h * 6.0 - 2.0)
    b = 2.0 - torch.abs(h * 6.0 - 4.0)
    return torch.clamp(i.vec3_const(r, g, b), 0.0, 1.0)


def hsl_to_rgb(hsl):
    i = IndexHelper(hsl)
    rgb = hue_to_rgb(i.idx(hsl, "x"))
    c = (1.0 - torch.abs(2.0 * i.idx(hsl, "z") - 1.0)) * i.idx(hsl, "y")
    return (rgb - 0.5) * c + i.idx(hsl, "z")


def rgb_to_hcv(rgb):
    i = IndexHelper(rgb)
    epsilon = 1e-5
    p = torch.where((i.idx(rgb, "y") < i.idx(rgb, "z")),
                    i.vec4_const(i.idx(rgb, "z"), i.idx(rgb, "y"), -1.0, 2.0 / 3.0),
                    i.vec4_const(i.idx(rgb, "y"), i.idx(rgb, "z"), 0.0, -1.0 / 3.0))

    q = torch.where((i.idx(rgb, "x") < i.idx(p, "x")),
                    i.vec4_const(i.idx(p, "x"), i.idx(p, "y"), i.idx(p, "w"), i.idx(rgb, "x")),
                    i.vec4_const(i.idx(rgb, "x"), i.idx(p, "y"), i.idx(p, "z"), i.idx(p, "x")))

    c = i.idx(q, "x") - torch.min(i.idx(q, "w"), i.idx(q, "y"))
    h = torch.abs((i.idx(q, "w") - i.idx(q, "y")) / (6.0 * c + epsilon) + i.idx(q, "z"))
    return i.vec3_const(h, c, i.idx(q, "x"))


def rgb_to_hsl(c):
    i = IndexHelper(c)
    epsilon = 1e-5

    hcv = rgb_to_hcv(c)
    l = i.idx(hcv, "z") - i.idx(hcv, "y") * 0.5
    s = i.idx(hcv, "y") / (1.0 - torch.abs(l * 2.0 - 1.0) + epsilon)
    return i.vec3_const(i.idx(hcv, "x"), s, l)


def rgb_to_hsv(c):
    i = IndexHelper(c)
    k = i.vec4_const(0, -1.0 / 3.0, 2.0 / 3.0, -1.0)

    condition = i.idx(c, "g") < i.idx(c, "b")

    xx = i.idx(c, "bg", debug=True)
    yy = i.idx(k, "wz")

    ca = i.cat(xx, yy)
    cb = i.cat(i.idx(c, "gb"), i.idx(k, "xy"))

    p = torch.where(condition, ca, cb)
    q = torch.where(i.idx(c, "r") < i.idx(p, "x"), i.cat(i.idx(p, "xyw"), i.idx(c, "r")),
                    i.cat(i.idx(c, "r"), i.idx(p, "yzx")))

    d = i.idx(q, "x") - i.min(i.idx(q, "w"), i.idx(q, "y"))
    e = 1e-10

    resx = torch.abs(i.idx(q, "z") + (i.idx(q, "w") - i.idx(q, "y")) / (6.0 * d + e))
    resy = d / (i.idx(q, "x") + e)
    resz = i.idx(q, "x")

    return i.vec3(resx, resy, resz)


def hsv_to_rgb(c):
    i = IndexHelper(c)
    K = i.vec4_const(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0)
    p = torch.abs(torch.frac(i.idx(c, "xxx") + i.idx(K, "xyz")) * 6.0 - i.idx(K, "www"))
    return i.idx(c, "z") * torch.lerp(i.idx(K, "xxx"), torch.clamp(p - i.idx(K, "xxx"), 0.0, 1.0), i.idx(c, "y"))


def xyz_to_lab(c):
    i = IndexHelper(c)
    n = c / i.vec3_const(95.047, 100, 108.883)
    v = i.vec3_const(0.0)

    i.set_idx(v, "x", torch.where(i.idx(n, "x") > 0.008856,
                                  torch.pow(torch.clamp(i.idx(n, "x"), min=0.008856), 0.333333),
                                  (7.787 * i.idx(n, "x")) + 0.137931))
    i.set_idx(v, "y", torch.where(i.idx(n, "y") > 0.008856,
                                  torch.pow(torch.clamp(i.idx(n, "y"), min=0.008856), 0.333333),
                                  (7.787 * i.idx(n, "y")) + 0.137931))
    i.set_idx(v, "z", torch.where(i.idx(n, "z") > 0.008856,
                                  torch.pow(torch.clamp(i.idx(n, "z"), min=0.008856), 0.333333),
                                  (7.787 * i.idx(n, "z")) + 0.137931))

    return i.vec3_const((116.0 * i.idx(v, "y")) - 16.0, 500.0 * (i.idx(v, "x") - i.idx(v, "y")),
                        200.0 * (i.idx(v, "y") - i.idx(v, "z")))


def xyz_to_rgb(c):
    i = IndexHelper(c)
    r = i.vec3_const(0.0)

    vx = i.idx(c, "x") * 3.2406 + i.idx(c, "y") * (-1.5372) + i.idx(c, "z") * (-0.4986)
    vy = i.idx(c, "x") * (-0.9689) + i.idx(c, "y") * 1.8758 + i.idx(c, "z") * 0.0415
    vz = i.idx(c, "x") * 0.0557 + i.idx(c, "y") * (-0.2040) + i.idx(c, "z") * 1.0570
    v = i.vec3(vx, vy, vz) * 0.01

    i.set_idx(r, "x", torch.where(i.idx(v, "r") > 0.0031308,
                                  (1.055 * torch.pow(torch.clamp(i.idx(v, "r"), min=0.0031308), 0.416667)) - 0.055,
                                  12.92 * i.idx(v, "r")))
    i.set_idx(r, "y", torch.where(i.idx(v, "g") > 0.0031308,
                                  (1.055 * torch.pow(torch.clamp(i.idx(v, "g"), min=0.0031308), 0.416667)) - 0.055,
                                  12.92 * i.idx(v, "g")))
    i.set_idx(r, "z", torch.where(i.idx(v, "b") > 0.0031308,
                                  (1.055 * torch.pow(torch.clamp(i.idx(v, "b"), min=0.0031308), 0.416667)) - 0.055,
                                  12.92 * i.idx(v, "b")))

    return r


def lab_to_xyz(c):
    i = IndexHelper(c)
    fy = (i.idx(c, "x") + 16.0) * 0.008620
    fx = i.idx(c, "y") * 0.002 + fy
    fz = fy - i.idx(c, "z") * 0.005

    return i.vec3_const(
        95.047 * torch.where(fx > 0.206897, fx * fx * fx, (fx - 16.0 * 0.008620) * 0.128419),
        100.000 * torch.where(fy > 0.206897, fy * fy * fy, (fy - 16.0 * 0.008620) * 0.128419),
        108.883 * torch.where(fz > 0.206897, fz * fz * fz, (fz - 16.0 * 0.008620) * 0.128419))


def rgb_to_xyz(c):
    i = IndexHelper(c)
    tmp = i.vec3_const(0.0)
    i.set_idx(tmp, "x", torch.where(i.idx(c, "r") > 0.04045,
                                    torch.pow((torch.clamp(i.idx(c, "r"), min=0.04045) + 0.055) * 0.947867, 2.4),
                                    i.idx(c, "r") * 0.077399))
    i.set_idx(tmp, "y", torch.where(i.idx(c, "g") > 0.04045,
                                    torch.pow((torch.clamp(i.idx(c, "g"), min=0.04045) + 0.055) * 0.947867, 2.4),
                                    i.idx(c, "g") * 0.077399))
    i.set_idx(tmp, "z", torch.where(i.idx(c, "b") > 0.04045,
                                    torch.pow((torch.clamp(i.idx(c, "b"), min=0.04045) + 0.055) * 0.947867, 2.4),
                                    i.idx(c, "b") * 0.077399))

    resx = i.idx(tmp, "x") * 0.4124 + i.idx(tmp, "y") * 0.3576 + i.idx(tmp, "z") * 0.1805
    resy = i.idx(tmp, "x") * 0.2126 + i.idx(tmp, "y") * 0.7152 + i.idx(tmp, "z") * 0.0722
    resz = i.idx(tmp, "x") * 0.0193 + i.idx(tmp, "y") * 0.1192 + i.idx(tmp, "z") * 0.9505

    return 100.0 * i.vec3(resx, resy, resz)


def rgb_to_lab(c):
    i = IndexHelper(c)
    lab = xyz_to_lab(rgb_to_xyz(c))
    return i.vec3(i.idx(lab, "x") * 0.01, 0.5 + i.idx(lab, "y") * 0.003937, 0.5 + i.idx(lab, "z") * 0.003937)


def lab_to_rgb(c):
    i = IndexHelper(c)
    return xyz_to_rgb(
        lab_to_xyz(
            i.vec3(100.0 * i.idx(c, "x"), 2.0 * 127.0 * (i.idx(c, "y") - 0.5), 2.0 * 127.0 * (i.idx(c, "z") - 0.5))))


def rgb_to_yuv(c):
    i = IndexHelper(c)
    r = i.idx(c, "r")
    g = i.idx(c, "g")
    b = i.idx(c, "b")

    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = 0.492 * (b - y) + 0.5
    v = 0.877 * (r - y) + 0.5

    v = torch.where(v < 1.0, v, torch.ones_like(v))
    return i.vec3(y, u, v)
