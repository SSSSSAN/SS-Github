import cv2
from matplotlib import pyplot as plt

path = 'test.jpg'
i = cv2.imread(path, 0)

for th in range(0,255,50):
    ret, i_binary = cv2.threshold(i, th, 255, cv2.THRESH_BINARY) 
    path_o = 'test_binary_th_'+ str(th) + '.jpg'
    cv2.imwrite(path_o, i_binary)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.imshow(i_binary, cmap = 'gray')
    ax1.tick_params(labelbottom = False, bottom = False)
    ax1.tick_params(labelleft = False, left = False)

    fig.tight_layout()
    plt.show()
    plt.close()