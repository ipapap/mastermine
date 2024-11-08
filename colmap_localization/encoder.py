import torch
import sys
sys.path.append('/home/gns/dev/gsv-cities/')
from main import VPRModel


from PIL import Image
import torchvision.transforms as T


MEAN=[0.485, 0.456, 0.406]; STD=[0.229, 0.224, 0.225]

IM_SIZE = (256, 256)

input_transform=T.Compose([
        # T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
		T.Resize(IM_SIZE,  interpolation=T.InterpolationMode.BILINEAR),
        
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])

def preprocess_image(image_path):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    # Preprocess the image
    image = input_transform(image)
    # Add batch dimension
    image = image.unsqueeze(0)
    return image
# define which device you'd like run experiments on (cuda:0 if you only have one gpu)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = VPRModel(
    
        #-------------------------------
        #---- Backbone architecture ----
        backbone_arch='myresnet_e2wrn18c4',#'myresnet_e2wrn',#'resnet50',  #'resnet50',#'myresnet_re_resnet50',  
        pretrained=True,
        layers_to_freeze=0,
        layers_to_crop=[], # 4 crops the last resnet layer, 3 crops the 3rd, ...etc
        
        #---------------------
        # ---- Aggregator -----
        # agg_arch='CosPlace',
        # agg_config={'in_dim': 800,#800,#2048,
        #             'out_dim': 512},#512},
        agg_arch='myAgg',
        agg_config={'in_channels': 312,
                    'out_channels': 512,
            },
    )
state_dict = torch.load('/home/gns/dev/gsv-cities/weights/myresnet_e2wrn18c4_epoch(26)_step(36477)_R1[0.6433]_R5[0.7637].ckpt') # link to the trained weights
# state_dict = torch.load('../ml-runs/986379816044858929/b6844bb810414cfeb2dee0e6152721d6/artifacts/model/checkpoints/resnet50_epoch(29)_step(40530)_R1[0.7052]_R5[0.8222]/resnet50_epoch(29)_step(40530)_R1[0.7052]_R5[0.8222].ckpt') # link to the trained weights

model.load_state_dict(state_dict['state_dict'])
model.eval()
model = model.to(device)



