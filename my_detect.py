import argparse
import copy
import os
import easyocr
import platform
import sys
import imutils
from pathlib import Path
sys.path.append("E:\WorkSpace\Python\Do_An_Tot_Nghiep\yolov9\my_WPOD")
sys.path.append("E:\WorkSpace\Python\Do_An_Tot_Nghiep\yolov9\my_WPOD\src")
from yolov9.my_WPOD import lp_detection , lp_recognition
from yolov9.my_WPOD import src
import torch


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


def check1(x1,k ):
    if x1 - k > 0 :
        return x1- k
    else :
        return 0
def check2(x1,w,k):
    if x1+k < w :
        return x1+k
    else:
        return w
def gia_anh(xyxy,img, m ):
    location = [int(x) for x in xyxy]
    folder_path_save = "E:\WorkSpace\Python\Do_An_Tot_Nghiep\Output"
    for i in range (0,30):
        name = "output_{}_{}".format(i,28)
        save_folder_path = os.path.join(folder_path_save,name)
        if os.path.exists(save_folder_path):
            continue
        else:
            os.makedirs(save_folder_path)
    x_bb = location[2] - location[0]
    y_bb = location[3] - location[1]
    h, w, _ = img.shape
    for i in range(0,30):
        x1 = int(check1(location[0],(i+1)*0.1*x_bb))
        y1 = int(check1(location[1],(28+1)*0.1*y_bb))

        x2= int(check2(location[2],w,(i+1)*0.1*x_bb))
        y2= int(check2(location[3],h,(28+1)*0.1*y_bb))

        crop_img = img[y1:y2,x1:x2]
        name_img = "img_{}.jpg".format(m)
        name = "output_{}_{}".format(i,28)
        path_save = os.path.join(folder_path_save,name,name_img)
        cv2.imwrite(path_save,crop_img)

def gia_anh_v2(xyxy,img, m ):
    location = [int(x) for x in xyxy]
    folder_path_save = "E:\WorkSpace\Python\Do_An_Tot_Nghiep\Output\output_-1"
    if not os.path.exists(folder_path_save):
        os.makedirs(folder_path_save)

    x1 = location[0]
    x2 = location[2]
    y1 = location [1]
    y2 = location[3]

    crop_img = img[y1:y2, x1:x2]
    name_img = "img_{}.jpg".format(m)
    path_save = os.path.join(folder_path_save, name_img)
    cv2.imwrite(path_save, crop_img)



def my_process_image(xyxy,model_wpod,model_reco,easyocr_model,img):
    """
    hàm xừ lý box từ yolov9 để đem sang wpod xoay
    :param xyxy: tọa độ xác định bbox của yolo
    :param model_wpod: model wpod để xử lý
    :param img: frame ảnh xác định
    :return: img_wpod
    """
    location = [int(x) for x in xyxy]
    default_str = " "
    k_x = 1.7  # hằng số mở rộng ảnh
    k_y = 2.9

    x_bb = location[2] - location[0]
    y_bb = location[3] - location[1]
    h, w, _ = img.shape

    x1 = int(check1(location[0], k_x * x_bb))
    y1 = int(check1(location[1], k_y * y_bb))

    x2 = int(check2(location[2], w, k_x  * x_bb))
    y2 = int(check2(location[3], h, k_y  * y_bb))

    img0 = img[y1:y2, x1:x2]
    if x_bb > 2.5 * y_bb:
        x1 = location[0]
        x2 = location[2]
        y1 = location[1]
        y2 = location[3]
        bbox = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(bbox, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 2)

        thresh = cv2.bitwise_not(thresh)
        thresh = imutils.resize(thresh, width=400)
        thresh = cv2.medianBlur(thresh, 5)
        output = easyocr_model.readtext(thresh)
        for out in output:
            text_bbox, text, text_score = out
            return text
    crop_img, img_wpod, img_copy = lp_detection.WPOD_detect(model_wpod, img0)
    if img_wpod is None:
        return default_str
    else:
        Lp_number = model_reco.predict(img_wpod)
        if Lp_number is None or len(Lp_number)<5:
            gray = cv2.cvtColor(img_wpod, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 2)

            thresh = cv2.bitwise_not(thresh)
            thresh = imutils.resize(thresh, width=400)
            thresh = cv2.medianBlur(thresh, 5)
            output = easyocr_model.readtext(thresh)
            res = ""
            for out in output:
                text_bbox, text, text_score = out
                res += text
            return res
        else:
            return Lp_number



def display_image(image_array):
    # Tính toán kích thước mới cho cửa sổ hiển thị
    new_width = int(image_array.shape[1] / 2)  # Kích thước mới theo chiều rộng
    new_height = int(image_array.shape[0] / 2)  # Kích thước mới theo chiều cao

    # Thay đổi kích thước cửa sổ hiển thị
    cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Resized Window', new_width, new_height)

    # Hiển thị ảnh trong cửa sổ mới
    cv2.imshow('Resized Window', image_array)
    cv2.waitKey(30)
@smart_inference_mode()
def run(
        weights=ROOT / 'yolo.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    my_detector = lp_detection.LicensePlateDetector()
    my_reco  = lp_recognition.LicensePlateRecognizer()
    easyocr_model = easyocr.Reader(['en'])
    m = 0
    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)


        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    #     with open(f'{txt_path}.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    # gia_anh_v2(xyxy,im0,m)
                    # m+=1

                    # print(xyxy)

                    lp_number = my_process_image(xyxy, my_detector,my_reco, easyocr_model ,im0)
                    label = lp_number

                    annotator.box_label(xyxy, label, color=colors(c, True))

                    # if save_crop:
                    #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                # cv2.imshow(str(p), im0)
                # cv2.waitKey(30)  # 1 millisecond
                display_image(im0)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


