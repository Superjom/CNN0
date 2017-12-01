import os
import cv2
import numpy as np


def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (0,0), fx=0.15, fy=0.15)
    return img

def write_image(img, path):
    cv2.imwrite(path, img)

pos_list_path = './people/pos.txt'
neg_list_path = './people/neg.txt'

pos_list = open(pos_list_path).read().split()
neg_list = open(neg_list_path).read().split()

counter = 0
path2label = []
data_meta = ''
for label, paths in enumerate([neg_list, pos_list]):
    for path in paths:
        try:
            img = load_image('./people/' + path)
            opath = os.path.join('./people.1', str(counter)+'.jpg')
            write_image(img, opath)
            counter += 1
            path2label.append((opath, label,))
        except:
            pass

with open('./people.1.txt', 'w') as f:
    content = ''
    for path, label in path2label:
        content += path + " " + str(label) + '\n'
    f.write(content)
