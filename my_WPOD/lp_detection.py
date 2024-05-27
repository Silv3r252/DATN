import cv2
import copy
import os
import traceback

import cv2
import numpy as np

from src.keras_utils import load_model, detect_lp
from src.utils2 import im2single


def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))


class LicensePlateDetector:
  def __init__(self, lp_threshold=.5, wpod_net_path = "E:\WorkSpace\Python\Do_An_Tot_Nghiep\yolov9\my_WPOD\weights\lp-detector\wpod-net_update1.h5"):
    self.lp_threshold = lp_threshold

    self.wpod_net = None
    self.load_model_wpod(wpod_net_path)
    
  def load_model_wpod(self, wpod_net_path):
    try:
      self.wpod_net = load_model(wpod_net_path)
    except:
      print('Can not load wpod net')
    # self.wpod_net = load_model(wpod_net_path)
  def detect(self, image):
    try:
      # print('Searching for license plates using WPOD-NET')
      # print('\t Processing')

      Ivehicle = image

      ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
      side  = int(ratio*288.)
      bound_dim = min(side + (side%(2**4)),608)
      # print("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))

      # Llp,LlpImgs,_ = detect_lp(self.wpod_net,im2single(Ivehicle),bound_dim,2**4,(120,100),self.lp_threshold)
      Llp, LlpImgs, _ = detect_lp(self.wpod_net, im2single(Ivehicle), bound_dim, 2 ** 4, (120,100) ,self.lp_threshold)
      if len(LlpImgs):
        Ilp = LlpImgs[0]
        Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
        Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

        # s = Shape(Llp[0].pts)
        # print(Llp[0])
        # cv2.imwrite('%s_lp.png' % (bname),Ilp*255.)
        # writeShapes('%s_lp.txt' % (bname),[s])

        return Ilp*255., Llp[0].pts

    except:
      traceback.print_exc()
    
    return None, None

def WPOD_detect(model,img):
  """
  Hàm giúp sử dụng WPOD Net để xoay ảnh

  :param model: model WPOD dùng để chuyển đổi
  :param img: ảnh dưới dạng ma trận
  :return: img_crop , img_wpod
  """
  img_copy = copy.deepcopy(img)
  lp_image,lp_labels = model.detect(img_copy)
  if lp_image is None:
    return  None, None , None

  polygon = list()
  h, w, _ = img.shape
  for i in range(len(lp_labels[0])):
    x = int(lp_labels[0][i] * img.shape[1])
    y = int(lp_labels[1][i] * img.shape[0])
    polygon.append([x, y])
  # vùng phát hiện lp  để xoay
  polygon = np.array(polygon, np.int32)

  cv2.polylines(img_copy, [polygon], isClosed=True, color=(0, 0, 255), thickness=1)
  x = sorted(polygon[:, 0])
  y = sorted(polygon[:, 1])
  for i in x :
    if i < 0 :
      i = 0
    elif i > w :
      i = w
    else:
      continue
  for i in y :
    if i < 0 :
      i = 0
    elif i > h :
      i = h
    else:
      continue

  crop_img = img_copy[y[0]:y[-1], x[0]:x[-1], :]
  # ảnh xoay
  img_wpod = np.round(lp_image).astype(np.uint8)

  return  crop_img,img_wpod, img_copy

# if __name__ == '__main__':
#
#   detector = LicensePlateDetector(.5, "weights/lp-detector/wpod-net_update1.h5")
#   data_path = os.path.join('E:\WorkSpace\Python\Do_An_Tot_Nghiep',"Data","img","03009.jpg")
#   img = cv2.imread(data_path)
#   crop_img,img_wpod = WPOD_detect(detector,img)
#   cv2.imshow("crop",crop_img)
#   cv2.imshow("wpod",img_wpod)
#   cv2.waitKey(0)
#
#
#   # sys.exit(0)


