"""inference fdlm+gz on image folder
"""
import os
import glob
import cv2
import numpy as np

from modules.scrfd import ScrfdFaceDetector

INT = True
DEBUG_INFO = False

if __name__ == "__main__":
    # model initial
    fd = ScrfdFaceDetector(int=INT, conf_thr=0.5, iou_thr=0.9, debug=DEBUG_INFO)
    # data initial
    image_pathes = glob.glob(os.path.join("Data", "example_images", "*.jpg"))
    print("---find {} images".format(len(image_pathes)))
    for image_path in image_pathes:
        image = cv2.imread(image_path)
        im2show = image.copy()
        # infernece
        bboxes, lms = fd.inference(image, debug=DEBUG_INFO)
        print("bboxes.shape: {}".format(bboxes.shape))
        if lms.shape[0] == 0:
            print("no face det: " + image_path)
            continue
        # plotter
        fd.plot(im2show, bboxes, lms, num=False)
        # show
        cv2.imshow("test", im2show)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
