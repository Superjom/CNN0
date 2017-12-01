import cv2
import numpy as np

batch_size = 30
test_size = 8

test_ratio = 0.1

# yield two numpy

def load_dataset(path):
    paths = open(path).read().split('\n')
    #np.random.shuffle(paths)

    images = []
    labels = []
    for no, path in enumerate(paths):
        fs = path.split()
        if len(fs) != 2: continue
        path, label = path.split()
        path = '.'+path
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
            labels.append([int(label)])
            #yield np.array(img), np.array([int(label)])

    images = np.array(images, dtype='float32')
    labels = np.array(labels)
    return images, labels


'''
def load_data(path_of_list):
    #TODO use a generator to save memory
    images, labels = load_dataset(path_of_list)

    train_size = int(images.shape[0] * (1-test_ratio))
    test_size = images.shape[0] - train_size

    train = images[:train_size], labels[:train_size]
    test = images[-test_size:], labels[-test_size:]
    return train, test
'''


if __name__ == '__main__':
    for batch in load_data('../people.1.txt'):
        print batch
