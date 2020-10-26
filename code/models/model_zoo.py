import segmentation_models_pytorch as smp
from torch import nn
from models.helper import Activation, DistanceMapHead, initialize_head, initialize_decoder
from models.baselineUnet.unet_model import UNet
from segmentation_models_pytorch.unet.decoder import UnetDecoder

aux_params=dict(
    pooling='avg',           # one of 'avg', 'max'
    dropout=0.5,            # dropout ratio, default is None
    activation='sigmoid',      # activation function, default is None
    classes=1,              # define number of output labels
)
in_channels=3
classes=1
encoder_weights='imagenet'
activation = None # 'sigmoid'

def baseline_unet(**kwargs):
    model = UNet(n_channels=in_channels, n_classes=classes, bilinear=False, **kwargs)
    print("baseline_unet",kwargs)
    return model

def resnet34_Unet(**kwargs):
    model = smp.Unet('resnet34', in_channels=in_channels, classes=classes, aux_params=aux_params, activation=activation, **kwargs)
    print("Segmentation + Classification Model args:")
    print("in_channels:%d,classes:%d,activation:%s" % (in_channels,classes,activation))
    print("aux_params:",aux_params,"kwargs",kwargs)
    return model

def resnet50_Unet(**kwargs):
    model = smp.Unet('resnet50', in_channels=in_channels, classes=classes, aux_params=aux_params, activation=activation, **kwargs)
    print("Segmentation + Classification Model args:")
    print("in_channels:%d,classes:%d,activation:%s" % (in_channels,classes,activation))
    print("aux_params:",aux_params,"kwargs",kwargs)
    return model

def efficientnet_b4_Unet(**kwargs):
    model = smp.Unet('efficientnet-b4', in_channels=in_channels, classes=classes, aux_params=aux_params, activation=activation, **kwargs)
    print("Segmentation + Classification Model args:")
    print("in_channels:%d,classes:%d,activation:%s" % (in_channels,classes,activation))
    print("aux_params:",aux_params,"kwargs",kwargs)
    return model

def resnet34_Unet_noclassification(**kwargs):
    model = smp.Unet('resnet34', in_channels=in_channels, classes=classes, activation=activation, **kwargs)
    print("Just Segmentation Model args:")
    print("in_channels:%d,classes:%d,activation:%s" % (in_channels,classes,activation))
    print("kwargs",kwargs)
    return model

def resnet50_Unet_noclassification(**kwargs):
    model = smp.Unet('resnet50', in_channels=in_channels, classes=classes, activation=activation, **kwargs)
    print("Just segmentation Model args:")
    print("in_channels:%d,classes:%d,activation:%s" % (in_channels,classes,activation))
    print("kwargs",kwargs)
    return model

def se_resnet50_Unet_noclassification(**kwargs):
    model = smp.Unet('se_resnext50_32x4d', in_channels=in_channels, classes=classes, activation=activation, **kwargs)
    print("Just segmentation Model args:")
    print("in_channels:%d,classes:%d,activation:%s" % (in_channels,classes,activation))
    print("kwargs",kwargs)
    return model

def resnet50_FPN_noclassification(**kwargs):
    model = smp.FPN('resnet50', in_channels=in_channels, classes=classes, activation=activation, **kwargs)
    print("Just segmentation Model args:")
    print("in_channels:%d,classes:%d,activation:%s" % (in_channels,classes,activation))
    print("kwargs",kwargs)
    return model

def resnet50_Linknet_noclassification(**kwargs):
    model = smp.Linknet('resnet50', in_channels=in_channels, classes=classes, activation=activation, **kwargs)
    print("Just segmentation Model args:")
    print("in_channels:%d,classes:%d,activation:%s" % (in_channels,classes,activation))
    print("kwargs",kwargs)
    return model

def resnet50_PSPNet_noclassification(**kwargs):
    model = smp.PSPNet('resnet50', in_channels=in_channels, classes=classes, activation=activation, **kwargs)
    print("Just segmentation Model args:")
    print("in_channels:%d,classes:%d,activation:%s" % (in_channels,classes,activation))
    print("kwargs",kwargs)
    return model

class Multihead_resnet50(nn.Module):

    def __init__(self, **kwargs):
        super(Multihead_resnet50, self).__init__()
        self.model = smp.Unet('resnet50', in_channels=in_channels, classes=classes, aux_params=aux_params, activation=activation,**kwargs)
        
        self.distancemap_head = DistanceMapHead(
            in_channels=16, # 原来此处为decoder_channels[-1]，而该参数默认值为decoder_channels: List[int] = (256, 128, 64, 32, 16)
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        initialize_head(self.distancemap_head)
        
    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.model.encoder(x)
        decoder_output = self.model.decoder(*features)

        masks = self.model.segmentation_head(decoder_output)
        distancemap = self.distancemap_head(decoder_output)

        labels = self.model.classification_head(features[-1])
        return masks, distancemap, labels
    
class Multihead_resnet50_noclassification(nn.Module):

    def __init__(self, **kwargs):
        super(Multihead_resnet50_noclassification, self).__init__()
        self.model = smp.Unet('resnet50', in_channels=in_channels, classes=classes, activation=activation, **kwargs)
        
        self.distancemap_head = DistanceMapHead(
            in_channels=16, # 原来此处为decoder_channels[-1]，而该参数默认值为decoder_channels: List[int] = (256, 128, 64, 32, 16)
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        initialize_head(self.distancemap_head)
        
    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.model.encoder(x)
        decoder_output = self.model.decoder(*features)

        masks = self.model.segmentation_head(decoder_output)
        distancemap = self.distancemap_head(decoder_output)
        return masks, distancemap
    
class Multibranch_resnet50(nn.Module):

    def __init__(self, **kwargs):
        super(Multibranch_resnet50, self).__init__()
        self.model = smp.Unet('resnet50', in_channels=in_channels, classes=classes, aux_params=aux_params, activation=activation,**kwargs)
        
        self.distancemap_branch = UnetDecoder(
            encoder_channels=self.model.encoder.out_channels,
            decoder_channels=(256, 128, 64, 32, 16),# List[int] = (256, 128, 64, 32, 16),
            n_blocks=5, # int = 5,
            use_batchnorm=True, # bool = True,
            center=False, #True if encoder_name.startswith("vgg") else False,
            attention_type=None, # Optional[str] = None,
        )
        self.distancemap_head = DistanceMapHead(
            in_channels=16, # 原来此处为decoder_channels[-1]，而该参数默认值为decoder_channels: List[int] = (256, 128, 64, 32, 16)
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        initialize_decoder(self.distancemap_branch)
        initialize_head(self.distancemap_head)
        
    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.model.encoder(x)
        decoder_mask_output = self.model.decoder(*features)
        decoder_distancemap_output = self.distancemap_branch(*features)

        masks = self.model.segmentation_head(decoder_mask_output)
        distancemap = self.distancemap_head(decoder_distancemap_output)
        
        labels = self.model.classification_head(features[-1])
        return masks, distancemap, labels
    
class Multibranch_resnet50_noclassification(nn.Module):

    def __init__(self, **kwargs):
        super(Multibranch_resnet50_noclassification, self).__init__()
        self.model = smp.Unet('resnet50', in_channels=in_channels, classes=classes, activation=activation, **kwargs)
        
        self.distancemap_branch = UnetDecoder(
            encoder_channels=self.model.encoder.out_channels,
            decoder_channels=(256, 128, 64, 32, 16),# List[int] = (256, 128, 64, 32, 16),
            n_blocks=5, # int = 5,
            use_batchnorm=True, # bool = True,
            center=False, #True if encoder_name.startswith("vgg") else False,
            attention_type=None, # Optional[str] = None,
        )
        self.distancemap_head = DistanceMapHead(
            in_channels=16, # 原来此处为decoder_channels[-1]，而该参数默认值为decoder_channels: List[int] = (256, 128, 64, 32, 16)
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        initialize_decoder(self.distancemap_branch)
        initialize_head(self.distancemap_head)
        
    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.model.encoder(x)
        decoder_mask_output = self.model.decoder(*features)
        decoder_distancemap_output = self.distancemap_branch(*features)

        masks = self.model.segmentation_head(decoder_mask_output)
        distancemap = self.distancemap_head(decoder_distancemap_output)
        return masks, distancemap