import torch
from torch.autograd import Variable
from rockethub import Rocket
from rocket_builder import preprocess
from darknet import YOLOv3
from PIL import Image

# --- LOAD IMAGE ---
# Select the image you want to test the Object Detection Model with
image_path = '/home/lucas/Documents/Mirage/rockethub-tutorial1/out.jpg'
img = Image.open(image_path)

# --- LOAD MODEL ---
model = YOLOv3()
model.eval()

# print(model)

# --- DETECTION ---
print('Using the rocket to do object detection on \'' + image_path + '\'...')
with torch.no_grad():
    img_tensor = preprocess(img)
    print(img_tensor.shape)
    out = model(img_tensor)

print('Object Detection successful! ')

print(out)
# --- OUTPUT ---
# Print the output as a JSON
# bboxes_out = model.postprocess(out, img)
# print(len(bboxes_out), 'different objects were detected:')
# print(*bboxes_out, sep='\n')

# # Display the output over the image
# img_out = model.postprocess(out, img, visualize=True)
# img_out_path = 'out.jpg'
# img_out.save(img_out_path)
# print('You can see the detections on the image: \'' + img_out_path +'\'.')

