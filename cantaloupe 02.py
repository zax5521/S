import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2 
import numpy as np
import win32gui
import win32con
import win32ui
import re
import serial.tools.list_ports
import time

CUSTOM_MODEL_NAME = 'cantaloupe_03_ssd640x640_4000step' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

previous_return_value = ""
return_value = ""

horizontal_detection_sensitive_value = 0.02  #數值越大越不敏感
Baud_rate = 115200  #鮑率
sec_to_detect = 3 #幾秒輸入
Detection_Scores = 0.7
portsList = []
boxwidth = []
nose_point = []
signal = "face_direction"
command = 0
x = 0

now_time = time.time()

ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-5')).expect_partial()
category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def FindWindow_bySearch(pattern):
    window_list = []
    win32gui.EnumWindows(lambda hWnd, param: param.append(hWnd), window_list)
    for each in window_list:
        if re.search(pattern, win32gui.GetWindowText(each)) is not None:
            return each

def getWindow_W_H(hwnd):
    # 取得目標視窗的大小
    left, top, right, bot = win32gui.GetWindowRect(hwnd)
    width = right - left - 15
    height = bot - top - 11
    return (left, top, width, height)

def getWindow_Img(hwnd):
    # 將 hwnd 換成 WindowLong
    s = win32gui.GetWindowLong(hwnd,win32con.GWL_EXSTYLE)
    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, s|win32con.WS_EX_LAYERED)
    # 判斷視窗是否最小化
    show = win32gui.IsIconic(hwnd)
    # 將視窗圖層屬性改變成透明    
    # 還原視窗並拉到最前方
    # 取消最大小化動畫
    # 取得視窗寬高
    if show == 1: 
        win32gui.SystemParametersInfo(win32con.SPI_SETANIMATION, 0)
        win32gui.SetLayeredWindowAttributes(hwnd, 0, 0, win32con.LWA_ALPHA)
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)    
        x, y, width, height = getWindow_W_H(hwnd)        
    # 創造輸出圖層
    hwindc = win32gui.GetWindowDC(hwnd)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    # 取得視窗寬高
    x, y, width, height = getWindow_W_H(hwnd)
    # 如果視窗最小化，則移到Z軸最下方
    if show == 1: win32gui.SetWindowPos(hwnd, win32con.HWND_BOTTOM, x, y, width, height, win32con.SWP_NOACTIVATE)
    # 複製目標圖層，貼上到 bmp
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0 , 0), (width, height), srcdc, (8, 3), win32con.SRCCOPY)
    # 將 bitmap 轉換成 np
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype = np.uint8)
    img.shape = (height, width, 4) #png，具有透明度的
    # 釋放device content
    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())
    # 還原目標屬性
    if show == 1 :
        win32gui.SetLayeredWindowAttributes(hwnd, 0, 255, win32con.LWA_ALPHA)
        win32gui.SystemParametersInfo(win32con.SPI_SETANIMATION, 1)
    # 回傳圖片
    return img

hwnd = FindWindow_bySearch("雷電模擬器")

for onePort in ports:
    portsList.append(str(onePort))
    print(str(onePort))

val = input(str("Select Port: COM"))
portVar = "COM" + str(val)
print(portVar)

serialInst.baudrate = Baud_rate
serialInst.port = portVar
serialInst.open()

while True: 
    frame = getWindow_Img(hwnd)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=Detection_Scores,
                agnostic_mode=False)

    for detection_boxes, detection_classes, detection_scores in \
        zip(detections['detection_boxes'], detections['detection_classes'], detections['detection_scores']):
        if detection_scores >= Detection_Scores and detection_classes == 0: #是哈密瓜
            #print(np.around(detection_boxes,4), detection_classes, round(detection_scores*100, 2))
            center_x_pos = (detection_boxes[1] + detection_boxes[3]) / 2
            bb = (detection_boxes[3] - detection_boxes[1])
            nose_point.append(center_x_pos)
            nose_point.append(center_x_pos)
            boxwidth.append(bb)
            max_width = max(boxwidth)
            max_width_num = boxwidth.index(max_width)
            nose_point_close_mid = min(nose_point, key = lambda x : abs(x - 0.5))
            nose_point_close_mid_num = nose_point.index(nose_point_close_mid)

    if len(boxwidth) > 1:
        st_dev = np.std(boxwidth)
        if st_dev >= horizontal_detection_sensitive_value:
          x = nose_point[max_width_num] - 0.5
        else:
          x = nose_point[nose_point_close_mid_num] - 0.5

    elif len(boxwidth) == 1:
      #nose_point[max_width_num] = nose_point[max_width_num].item() #把numpy.float32 轉成 float
      x = nose_point[max_width_num] - 0.5
    
    else:
      x = 0
      angle = 0

    if x > 0:  #畫面右邊
      # = 16.281x4 - 28.017x3 + 3.98x2 + 52.984x + 0.0091
      angle =  16.281 * pow(x, 4) - 28.017 * pow(x, 3) + 3.98 * pow(x, 2) + 52.984 * x + 0.0091
    elif x < 0: #畫面左邊
      x = abs(x)
      angle = 16.281 * pow(x, 4) - 28.017 * pow(x, 3) + 3.98 * pow(x, 2) + 52.984 * x + 0.0091
      angle = -angle
    
    #在畫面中間畫一條線
    windows_width = image_np_with_detections.shape[1]
    windows_height = image_np_with_detections.shape[0]
    image_np_with_detections = cv2.line(image_np_with_detections, (int(windows_width/2), 0), (int(windows_width/2), windows_height), (0, 0 ,225), 1) #參數中的int是因為畫線不給float

    #接收回傳的步數 看有沒有符合一致
    while serialInst.in_waiting:
      received_data = serialInst.read().strip()
      for i in str(received_data):
         if i != 'b' and i != '\'':
            return_value += i
    if previous_return_value != return_value:
      return_value = return_value.replace(return_value[0:len(previous_return_value)], "")
    if serialInst.in_waiting == False:
      previous_return_value = return_value

    #在畫面左上顯示角度和步數
    command = round(angle/(1.8/16))  #計算要走幾步
    text = "angle = " + str(angle) + "    x_pos = " + str(x) +"     command = " + str(command)
    cv2.putText(image_np_with_detections, text, (20, 50),cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
    text = "return_step:  " + str(return_value)
    cv2.putText(image_np_with_detections, text, (20, 90),cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)

    #每sec_to_detect偵測一次
    if time.time() - now_time >= sec_to_detect:
      now_time = time.time()
      #print(f"angle = {angle:>20}    x_pos = {x:< 30}   command = {command:> 4}")
      angle = str(angle)
      command = str(command)
      serialInst.write(command.encode('utf-8'))
    boxwidth.clear()
    nose_point.clear()

    cv2.imshow('object detection',  image_np_with_detections)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
exit()
