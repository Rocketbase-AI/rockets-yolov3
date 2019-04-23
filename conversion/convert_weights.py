import torch
import csv
# from ..darknet import YOLOv3

# import the layers correspondance
list_layers = []
with open('conversion/YOLOv3-Layer_Correspondance.csv', 'r') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        list_layers.append(row)

# convert list to dictionary
dict_layers = {i[0]:i[1] for i in list_layers}

# import the original dictionary
weights = torch.load('weights.pth')

new_weights = {}

for key, value in weights.items():
    # Split the key name
    current_layer = key[:key.rfind('.')]
    current_weight = key[key.rfind('.'):]

    if current_layer in dict_layers.keys():
        new_key = dict_layers[current_layer] + current_weight
        # weights[new_key] = weights.pop(key)
        new_weights[new_key] = weights[key]

    else:
        print('missing:', current_layer)

for key, value in new_weights.items():
    print(key) 

torch.save(new_weights, 'new_weights.pth')

# model = YOLOv3()
# model.load_state_dict(torch.load('weights.pth'), strict=False)
# model.eval()