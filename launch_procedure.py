import os
import shutil
import json
from rocketbase import Rocket

username = 'lucas'
modelName = 'yolov3'
hash = 'nextR'

rockets_path = 'rockets'
next_rocket_path = os.path.join(rockets_path, '{username}_{modelName}_{hash}')

# Land the previous rocket
_ = Rocket.land('{username}/{modelName}')

# Get the name of the folder where the Rocket just landed
old_rocket_folder = [f for f in os.listdir(rockets_path) if not f.startswith('.')][0]

# create the folder to prepare the rocket
os.mkdir(next_rocket_path)
shutil.move(os.path.join(old_rocket_folder, 'weights.pth'), os.path.join(next_rocket_path, 'weights.pth'))

# copy all the file needed from the repository in the new folder
with open('info.json', r) as f:
    info = json.load(f)

for f in info['blueprint']:
    shutil.move(f, os.path.join(next_rocket_path, f))

# Launch the new Rocket
is_launch_successful = Rocket.launch('{username}_{modelName}_{hash}')

