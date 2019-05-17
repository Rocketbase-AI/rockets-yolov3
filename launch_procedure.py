import os
import shutil
import sys
import json
from rocketbase import Rocket

username = 'lucas'
modelName = 'yolov3'
rocket_hash = 'next'

rocket_folder_name = '{}_{}_{}'.format(username, modelName, rocket_hash)

rocket_slug = '{}/{}'.format(username, modelName)

rockets_path = 'rockets'
next_rocket_path = os.path.join(rockets_path, rocket_folder_name)

# Land the previous rocket
_ = Rocket.land(rocket_slug)

# Get the name of the folder where the Rocket just landed
old_rocket_folder = [f for f in os.listdir(rockets_path) if not f.startswith('.') and not f == 'rockets'][0]

old_rocket_folder = os.path.join(rockets_path, old_rocket_folder)

# create the folder to prepare the rocket
os.mkdir(next_rocket_path)
shutil.move(os.path.join(old_rocket_folder, 'weights.pth'), os.path.join(next_rocket_path, 'weights.pth'))

# copy all the file needed from the repository in the new folder
with open('info.json', 'r') as f:
    info = json.load(f)

for f in info['blueprint']:
    if not f == 'weights.pth':
        shutil.move(f, os.path.join(next_rocket_path, f))

# Launch the new Rocket
is_launch_successful = Rocket.launch(rocket_slug + '/' + rocket_hash)

sys.exit(0)

