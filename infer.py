from models.options import Options
opt = Options().paser()
from models.gargs import _args
import cv2
import os
from models.acdnet.acdnet import ACDNet
from models.utils import *

model = ACDNet().cuda();model.eval()
# print(model)
state_dict = torch.load(_args['checkpoints'],map_location='cpu')
print('Load checkpoints from {} ...'.format(_args['checkpoints']))
model.load_state_dict(state_dict['model'],strict=True)
print('Load image from {} ...'.format(_args['example']))
example = load_imgs(_args['example']).cuda()
with torch.no_grad():
    depth = model(example)['depth']
fdir,fname = os.path.dirname(_args['example']),os.path.basename(_args['example']).split('.')[0]
depth = tensor2img(depth[0,0])
print('Save image to {} ...'.format(os.path.join(fdir,fname+'_pred.png')))
cv2.imwrite(os.path.join(fdir,fname+'_pred.png'),depth)
