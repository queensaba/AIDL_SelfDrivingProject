import json

with open('c:/ceci/aidl_project/AIDL_SelfDrivingProject/data/transfer_learning/bdd100k_labels_images_train.json') as f:
    data = json.load(f)

file = open('c:/ceci/aidl_project/AIDL_SelfDrivingProject/data/transfer_learning/val/val.txt', "r") 
file_lines = file.read()
images = file_lines.split("\n")

output_dict = [x for x in data if x['name'] in images]
output_json = json.dumps(output_dict)

with open('c:/ceci/aidl_project/AIDL_SelfDrivingProject/data/transfer_learning/val/labels_TL_val.json','w') as outfile:
    outfile.write(output_json)