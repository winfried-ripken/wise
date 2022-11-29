import torch
from torch import Tensor

from helpers.custom_functions import opengl_round


class IndexHelper:
    def __init__(self, image: torch.Tensor):
        self.default_shape = image.shape[0:1] + (1,) + image.shape[2:]
        self.device = image.device

    @staticmethod
    def complete(result):
        if len(result.size()) == 3:
            result = result.unsqueeze(1)

        if result.size(1) == 1:
            return IndexHelper.cat(result, torch.zeros((result.shape[0], 2) + result.shape[2:], device=result.device))
        elif result.size(1) == 2:
            return IndexHelper.cat(result, torch.zeros((result.shape[0], 1) + result.shape[2:], device=result.device))
        else:
            return result

    @staticmethod
    def same_img_size(*tensors: torch.tensor):
        img_shape = tensors[0].shape[2:]
        for tensor in tensors:
            if img_shape != tensor.shape[2:]:
                return False
        return True


    @staticmethod
    def generate_result(result):
        return opengl_round(result)

    def const(self, value: float):
        return torch.ones(self.default_shape, device=self.device) * value

    @staticmethod
    def view(value: Tensor, dims=4):
        if len(value.size()) <= 1:
            # single value or list
            if dims == 5:
                return value.view(-1, 1, 1, 1, 1)
            if dims == 4:
                return value.view(-1, 1, 1, 1)
            elif dims == 3:
                return value.view(-1, 1, 1)
        if len(value.size()) == 2:
            # value has shape BSx1 or BSx3
            if dims == 5:
                return value.view(-1, value.size(1), 1, 1, 1)
            if dims == 4:
                return value.view(-1, value.size(1), 1, 1)
            elif dims == 3:
                return value.view(-1, value.size(1), 1)
        elif len(value.size()) == 3:
            if dims == 5:
                return value.unsqueeze(1).unsqueeze(1)
            if dims == 4:
                return value.unsqueeze(1)
            if dims == 3:
                return value
        elif len(value.size()) == 4:
            # this is a localized parameter mask
            # dims are BS x 1 x H x W or BS x 3 x H x W
            if dims == 5:
                return value.unsqueeze(1)  # this highly depends on where we add the additional dimension
                # we should have a convention for this, but unfortunately we don't
            if dims == 4:
                return value
            if dims == 3:
                return value.squeeze(1)

    @staticmethod
    def dot(a, b):
        return (a * b).sum(dim=1, keepdim=True)

    @staticmethod
    def cat(*tensors):
        return torch.cat(tensors, dim=1)

    @staticmethod
    def cross(a, b):
        return torch.cross(a, b, dim=1)

    @staticmethod
    def mat2(a00, a01, a10, a11):
        return IndexHelper.cat(a00, a01, a10, a11)

    def mat2_vec2_prod(self, m, v):
        m00 = IndexHelper.mx(m, 0, 0)
        m10 = IndexHelper.mx(m, 0, 1)
        m01 = IndexHelper.mx(m, 1, 0)
        m11 = IndexHelper.mx(m, 1, 1)

        v0 = IndexHelper.idx(v, "x")
        v1 = IndexHelper.idx(v, "y")

        return self.vec2(m00 * v0 + m10 * v1,
                         m01 * v0 + m11 * v1)

    @staticmethod
    def mat2_prod(a, b):
        a00 = IndexHelper.mx(a, 0, 0)
        a10 = IndexHelper.mx(a, 0, 1)
        a01 = IndexHelper.mx(a, 1, 0)
        a11 = IndexHelper.mx(a, 1, 1)

        b00 = IndexHelper.mx(b, 0, 0)
        b10 = IndexHelper.mx(b, 0, 1)
        b01 = IndexHelper.mx(b, 1, 0)
        b11 = IndexHelper.mx(b, 1, 1)

        return IndexHelper.mat2(a00 * b00 + a01 * b10,
                                a00 * b01 + a01 * b11,
                                a10 * b00 + a11 * b10,
                                a10 * b01 + a11 * b11)

    def mat_vec_prod(self, m, v):
        # matrix and vector should both be localized
        # (i.e. contain spatial dimensions)

        # M: B x (K x K) x H x W
        # V: B x K x H x W

        v_dim = v.size(1)
        m = m.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)

        # openGL matrices have rows and cols swapped
        m = m.reshape(m.size(0), m.size(1), m.size(2), v_dim, v_dim).permute(0, 1, 2, 4, 3)
        v = v.view(v.size(0), v.size(1), v.size(2), 1, v_dim)

        return torch.matmul(v, m).squeeze(-2).permute(0, -1, 1, 2)

    @staticmethod
    def mx(tensor: torch.Tensor, i1, i2):
        idx = i1 * 2 + i2
        return tensor[:, idx:idx + 1]

    def min(self, *tensors):
        return self.cat(*tensors).min(dim=1, keepdim=True).values

    def input_size_factor(self):
        h, w = self.default_shape[2:]
        return min(w, h) / 512.0

    def tex_size(self, hw="wh"):  # NOTE: inverted by default
        h, w = self.default_shape[2:]

        if hw == "wh":
            return torch.tensor([w, h], device=self.device).view(1, 2, 1, 1).repeat(self.default_shape[0], 1, 1,
                                                                                    1).float()
        elif hw == "hw":
            return torch.tensor([h, w], device=self.device).view(1, 2, 1, 1).repeat(self.default_shape[0], 1, 1,
                                                                                    1).float()
        elif hw == "h":
            return torch.tensor([h], device=self.device).view(1, 1, 1, 1).repeat(self.default_shape[0], 1, 1, 1).float()
        elif hw == "w":
            return torch.tensor([w], device=self.device).view(1, 1, 1, 1).repeat(self.default_shape[0], 1, 1, 1).float()
        else:
            raise ValueError(f"{hw} not recognized")

    @staticmethod
    def smoothstep(edge0, edge1, x):
        t = torch.clamp(IndexHelper.safe_div((x - edge0), (edge1 - edge0)), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def get_len(x, dim=1, keepdim=True):
        # add epsilon because the gradient of sqrt(0) is undefined
        return torch.sqrt(torch.square(x).sum(dim=dim, keepdim=keepdim) + torch.finfo(x.dtype).eps)

    @staticmethod
    def get_rgb_and_alpha(x):
        if x.size(1) == 4:
            return IndexHelper.idx(x, "rgb"), IndexHelper.idx(x, "a")
        else:
            return x, None

    @staticmethod
    def combine_rgb_and_alpha(x, alpha):
        if alpha is None:
            return x
        else:
            return IndexHelper.cat(x, alpha)

    @staticmethod
    def idx(tensor: torch.Tensor, args, debug=False):
        result = []

        for a in args:
            if a == "x":
                result.append(tensor[:, 0:1])
            elif a == "y":
                result.append(tensor[:, 1:2])
            elif a == "z":
                result.append(tensor[:, 2:3])
            elif a == "w":
                result.append(tensor[:, 3:4])
            elif a == "r":
                result.append(tensor[:, 0:1])
            elif a == "g":
                result.append(tensor[:, 1:2])
            elif a == "b":
                result.append(tensor[:, 2:3])
            elif a == "a":
                result.append(tensor[:, 3:4])
            elif a == "s":
                result.append(tensor[:, 0:1])
            elif a == "t":
                result.append(tensor[:, 1:2])
            # these are my additions, fill with ones or zeros for debugging
            elif a == "n":
                result.append(torch.zeros_like(tensor[:, 0:1]))
            elif a == "p":
                result.append(torch.ones_like(tensor[:, 0:1]))

        return IndexHelper.cat(*result)

    @staticmethod
    def set_idx(tensor: torch.Tensor, args, value: torch.Tensor):
        for i, a in enumerate(args):
            if a == "x":
                tensor[:, 0:1] = value[:, i:i + 1]
            elif a == "y":
                tensor[:, 1:2] = value[:, i:i + 1]
            elif a == "z":
                tensor[:, 2:3] = value[:, i:i + 1]
            elif a == "w":
                tensor[:, 3:4] = value[:, i:i + 1]
            elif a == "r":
                tensor[:, 0:1] = value[:, i:i + 1]
            elif a == "g":
                tensor[:, 1:2] = value[:, i:i + 1]
            elif a == "b":
                tensor[:, 2:3] = value[:, i:i + 1]
            elif a == "s":
                tensor[:, 0:1] = value[:, i:i + 1]
            elif a == "t":
                tensor[:, 1:2] = value[:, i:i + 1]

    def vec2(self, x, y):
        return self.cat(x, y)

    def vec2r(self, x):
        return self.cat(x, x)

    def vec2_const(self, x, y=None):
        if y is None:
            y = x

        x = self.const(x)
        y = self.const(y)

        return self.vec2(x, y)

    def vec3(self, x, y, z):
        return self.cat(x, y, z)

    def vec3r(self, x):
        return self.cat(x, x, x)

    def vec3_const(self, x, y=None, z=None):
        if y is None:
            y = x
        if z is None:
            z = y

        x = self.const(x)
        y = self.const(y)
        z = self.const(z)

        return self.vec3(x, y, z)

    def vec4(self, x, y, z, w):
        return self.cat(x, y, z, w)

    def vec4r(self, x):
        return self.cat(x, x, x, x)

    def vec4_const(self, x, y=None, z=None, w=None):
        if y is None:
            y = x
        if z is None:
            z = y
        if w is None:
            w = z

        x = self.const(x)
        y = self.const(y)
        z = self.const(z)
        w = self.const(w)

        return self.vec4(x, y, z, w)

    # this is the same like v_TexCoord in OpenGL
    def get_p_start(self):
        batch_size, _, image_height, image_width = self.default_shape

        pixel_row = torch.arange(0, image_height, device=self.device) + 0.5
        pixel_col = torch.arange(0, image_width, device=self.device) + 0.5

        yy, xx = torch.meshgrid(pixel_row, pixel_col)
        xx = xx.float()
        yy = yy.float()
        # keep range same as opengl
        # In OpenGL the Y Axis is inverted
        xx = xx / image_width
        yy = yy / image_height
        yy = yy.flip(dims=[0])

        # adding (X,Y) spatial dimension
        # adding batch dim
        return torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)

    # this is the same like texture2D(...) in OpenGL
    def sample(self, x, grid, mode="bilinear", padding_mode="border", add_p_start=False,
               invert_y=True):
        # Range is from [-1,1] unlike OpenGL!
        # scale grid to [-1,1]
        # Note that invert_y should always be on
        # We need this to sidestep a "bug" in Arctic

        if add_p_start:
            p = self.get_p_start() + grid
        else:
            p = grid

        # allow to sample static textures
        if self.default_shape[0] > x.size(0):
            x = x.repeat(self.default_shape[0], 1, 1, 1)

        vgrid_x = 2.0 * p[:, 0, ...] - 1.0
        vgrid_y = 2.0 * (1.0 - p[:, 1, ...]) - 1.0 if invert_y else 2.0 * p[:, 1, ...] - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)

        if padding_mode == "repeat":
            padding_mode = "border"
            vgrid_scaled = vgrid_scaled + 1
            vgrid_scaled = torch.remainder(vgrid_scaled, 2)
            vgrid_scaled = vgrid_scaled - 1

        return torch.nn.functional.grid_sample(x, vgrid_scaled, mode=mode, padding_mode=padding_mode,
                                               align_corners=False)

    # this is the same like texture3D(...) in OpenGL
    # x needs to have dims BxCxDxHxW
    def sample_3d(self, x, p, mode="bilinear", padding_mode="border"):
        vgrid_x = 2.0 * p[:, 0, ...] - 1.0
        vgrid_y = 2.0 * p[:, 1, ...] - 1.0
        vgrid_d = 2.0 * p[:, 2, ...] - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y, vgrid_d), dim=4)

        return torch.nn.functional.grid_sample(x, vgrid_scaled, mode=mode, align_corners=False,
                                               padding_mode=padding_mode)

    # samples texture multiple times by iterating axis 1
    def sample_complex(self, x, grid, mode="bilinear", add_p_start=False):
        bs = grid.shape[0]
        n_samples = grid.shape[1]

        if add_p_start:
            p = self.get_p_start().unsqueeze(1) + grid
        else:
            p = grid

        p = p.reshape(bs * n_samples, *p.shape[2:])
        s = self.sample(x.repeat_interleave(n_samples, dim=0), p, mode=mode)

        return s.reshape(bs, n_samples, *s.shape[1:])

    @staticmethod
    def safe_tensor(x, eps=None):
        if eps is None:
            if type(x) == float:
                eps = 1e-7
            else:
                eps = torch.finfo(x.dtype).eps * 2

        if type(x) == float:
            if x > eps:
                return x
            elif x < -eps:
                return x
            elif x > 0:
                return eps
            else:
                return -eps

        xcl_pos = torch.clamp(x, min=eps)
        xcl_neg = torch.clamp(x, max=-eps)
        return torch.where(x < 0, xcl_neg, xcl_pos)

    @staticmethod
    def safe_div(x, y):
        return x / IndexHelper.safe_tensor(y)

    @staticmethod
    def safe_pow(x, y):
        # if x is negative, pow is undefined in OpenGL
        x = torch.clamp(x, min=torch.finfo(x.dtype).eps * 2)
        return torch.pow(x, y)

    @staticmethod
    def safe_sqrt(x):
        x = torch.clamp(x, min=torch.finfo(x.dtype).eps * 2)
        return torch.sqrt(x)

    @staticmethod
    def safe_abs(x):
        return torch.abs(IndexHelper.safe_tensor(x))

    @staticmethod
    def old_style_clamp(x, mi, ma):
        x = torch.where(x < mi, mi, x)
        x = torch.where(x > ma, ma, x)
        return x


class OpenGLProcessor:
    @staticmethod
    def convert_line(line):
        result = ""

        c_old = ""
        indexing = False

        for c in line:
            if c == ";":
                continue

            if c in ["x", "y", "z", "w", "r", "g", "b"]:
                if c_old == ".":
                    indexing = True
                    last_c = result[-2]
                    result = result[:-2]
                    result += "i.idx("
                    result += last_c
                    result += ', "'
            else:
                if indexing:
                    result += '")'
                    indexing = False

            result += c
            c_old = c

        return result

    def convert_text(self, ogl_string):
        print("i = IndexHelper(c)")
        for line in ogl_string.split("\n"):
            line = self.convert_line(line)

            line = line.replace("//", "# ")
            line = line.replace("vec2 ", "")
            line = line.replace("vec3 ", "")
            line = line.replace("vec4 ", "")
            line = line.replace("float ", "")

            line = line.replace("vec2(", "i.vec2_const(")
            line = line.replace("vec3(", "i.vec3_const(")
            line = line.replace("vec4(", "i.vec4_const(")

            line = line.replace("atan", "torch.atan")
            line = line.replace("pow", "torch.pow")
            line = line.replace("mix", "torch.lerp")
            line = line.replace("abs", "torch.abs")
            line = line.replace("clamp", "torch.clamp")
            line = line.replace("fract", "torch.frac")
            print(line)


if __name__ == '__main__':
    with open("glsnippet.txt") as f:
        OpenGLProcessor().convert_text(f.read())
