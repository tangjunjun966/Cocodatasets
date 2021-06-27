from .loadfromimgfile import LoadImageFromFile
from .loadannolations import LoadAnnotations
from .resize import Resize
from .randomflip import RandomFlip
from .randomcrop import RandomCrop
from .randomcropresize import RandomCropResize

from .pad import Pad
from .normalize import Normalize
from .defaultformatbundle import DefaultFormatBundle
from .collect import Collect

pipelines_cls = {'LoadImageFromFile': LoadImageFromFile,
                 'LoadAnnotations': LoadAnnotations,
                 'Resize': Resize,
                 'RandomFlip': RandomFlip,
                 'RandomCrop': RandomCrop,
                 'RandomCropResize': RandomCropResize,

                 'Normalize': Normalize,
                 'Pad': Pad,
                 'DefaultFormatBundle': DefaultFormatBundle,
                 'Collect': Collect

                 }
