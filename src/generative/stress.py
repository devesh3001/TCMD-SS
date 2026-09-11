"""Deterministic development corruptions; all source images remain normal terrain."""
import hashlib
import math
import random
from PIL import Image, ImageFilter
import torch

FAMILIES = ('stripes', 'dead_pixels', 'missing_patch', 'local_blur', 'patch_duplication', 'foreign_content')


def tensor_image(image):
    return torch.tensor(list(image.convert('L').get_flattened_data()), dtype=torch.float32).reshape(1,image.height,image.width)/255


def pil_image(values):
    values=values.detach().cpu().squeeze()
    return Image.frombytes('L',(values.shape[1],values.shape[0]),bytes(values.clamp(0,1).mul(255).round().to(torch.uint8).flatten().tolist()))


def stress(image, family, severity, seed):
    if family not in FAMILIES or severity not in (.15,.30,.50):
        raise ValueError('This development protocol has six families and three fixed severities')
    rng=random.Random(seed)
    image=image.convert('L')
    w,h=image.size
    pixels=list(image.get_flattened_data()); result=pixels.copy(); mask=[0]*len(pixels)
    side=max(4,int(min(w,h)*(.08+.35*severity)))
    # Keep spatial seeds shared across severities so strength changes are easier to interpret.
    x=rng.randrange(max(1,w//2)); y=rng.randrange(max(1,h//2))
    if family=='stripes':
        horizontal=bool(seed%2);phase=rng.random()*2*math.pi
        for j in range(h):
            for i in range(w):
                k=j*w+i
                delta=255*.2*severity*math.sin(2*math.pi*(j if horizontal else i)/8+phase)
                result[k]=max(0,min(255,round(pixels[k]+delta))); mask[k]=255
    elif family=='dead_pixels':
        for k in range(len(pixels)):
            if rng.random()<.002+.02*severity:
                result[k]=255 if rng.randrange(2) else 0; mask[k]=255
    elif family=='local_blur':
        side=int(min(w,h)*(.25+.3*severity))
        blurred=list(image.filter(ImageFilter.GaussianBlur(.5+5*severity)).get_flattened_data())
        for j in range(y,min(h,y+side)):
            for i in range(x,min(w,x+side)):
                k=j*w+i;result[k]=blurred[k];mask[k]=255
    else:
        if family=='patch_duplication':
            # Copy between separated quadrants; overlapping source/target can create no-ops.
            x=w-side-2;y=h-side-2
        for j in range(side):
            for i in range(side):
                k=(y+j)*w+x+i
                if family=='missing_patch': value=0
                elif family=='patch_duplication': value=pixels[(2+j)*w+2+i]
                else:
                    pattern=255 if ((i//3+j//3)%2) else 0
                    alpha=.3+.7*severity;value=round(pixels[k]*(1-alpha)+pattern*alpha)
                result[k]=value;mask[k]=255
    changed=sum(a!=b for a,b in zip(pixels,result))
    if not changed:
        raise ValueError('Corruption made no visible change; retain as explicit benchmark failure')
    output=Image.frombytes('L',(w,h),bytes(result))
    return output, Image.frombytes('L',(w,h),bytes(mask))


def case_seed(base_id, family):
    return int(hashlib.sha256(f'42:{base_id}:{family}'.encode()).hexdigest()[:8],16)
