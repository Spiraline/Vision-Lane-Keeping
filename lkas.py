#!/usr/bin/env python3
import numpy as np
import rospy
import cv2
import math
import yaml
from enum import Enum, auto

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import TwistStamped

X_M_PER_PIX = 3.7/70
Y_M_PER_PIX = 30/720 # meters per pixel in y dimension

class turnState(Enum):
    FORWARD = auto()
    LEFT = auto()
    RIGHT = auto()

def find_firstfit(binary_array, direction, lower_threshold=35):
    # direction : -1 (right to left), 1 (left to right)
    if direction == 1:
        start, end = (0, binary_array.shape[0])
        for i in range(start, end-3, direction):
            # If value is larger than threshold, it is white image
            if binary_array[i]+binary_array[i+1]+binary_array[i+2]+binary_array[i+3]> lower_threshold:
                return i
    else:
        start, end = (binary_array.shape[0]-1, -1)
        for i in range(start, end+3, direction):
            # If value is larger than threshold, it is white image
            if binary_array[i]+binary_array[i-1]+binary_array[i-2]+binary_array[i-3]> lower_threshold:
                return i
    return -1

def color_gradient_filter(img, filter_thr_dict):
    s_thresh = filter_thr_dict['saturation_thr']
    sx_thresh = filter_thr_dict['x_grad_thr']
    r_thresh = filter_thr_dict['red_thr']
    g_thresh = filter_thr_dict['green_thr']
    b_thresh = filter_thr_dict['blue_thr']
    img = np.copy(img)
    
    R = img[:,:,2]
    G = img[:,:,1]
    B = img[:,:,0]
    
    # Threshold red channel
    rbinary = np.zeros_like(R)
    rbinary2 = np.zeros_like(R)
    rbinary[(R >= r_thresh[0]) & (R <= r_thresh[1])] = 1
    rbinary2[(R >= 205) & (R <= 250)] = 1

    gbinary = np.zeros_like(G)
    gbinary2 = np.zeros_like(G)
    gbinary[(G >= g_thresh[0]) & (G <= g_thresh[1])] = 1
    gbinary2[(G >= 205) & (G <= 250)] = 1

    bbinary = np.zeros_like(B)
    bbinary2 = np.zeros_like(B)    
    bbinary[(B >= b_thresh[0]) & (B <= b_thresh[1])] = 1
    bbinary2[(B >= 195) & (B <= 250)] = 1

    # Convert to HLS color space and separate the s channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)#.astype(np.float)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx)) #cv2.convertScaleAbs(abs_sobelx)

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold saturation channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    combined_binary = np.zeros_like(sxbinary)
    # combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    # combined_binary[(s_binary == 1) | (sxbinary == 1) & (rbinary == 1)] = 1
    
    combined_binary[(((rbinary == 1) & (gbinary == 1) & (bbinary == 1)) | ((rbinary2 == 1) & (gbinary2 == 1) & (bbinary2 == 1)))] = 1
    
    return combined_binary

def birdeye_warp(img, birdeye_warp_param):
    x_size = img.shape[1]
    y_size = img.shape[0]
    x_mid = x_size/2

    # Tunable Parameter
    top = birdeye_warp_param['top']
    bottom = birdeye_warp_param['bottom']
    top_margin = birdeye_warp_param['top_width']
    bottom_margin = birdeye_warp_param['bottom_width']
    birdeye_margin = birdeye_warp_param['birdeye_width']

    # 4 Source coordinates
    src1 = [x_mid + top_margin, top] # top_right
    src2 = [x_mid + bottom_margin, bottom] # bottom_right
    src3 = [x_mid - bottom_margin, bottom] # bottom_left
    src4 = [x_mid - top_margin, top] # top_left
    src = np.float32([src1, src2, src3, src4])

    # 4 destination coordinates
    dst1 = [x_mid + birdeye_margin, 0]
    dst2 = [x_mid + birdeye_margin, y_size]
    dst3 = [x_mid - birdeye_margin, y_size]
    dst4 = [x_mid - birdeye_margin, 0]
    dst = np.float32([dst1, dst2, dst3, dst4])

    # Given src and dst points, calculate the perspective transform matrix
    trans_matrix = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    img_size = (img.shape[1], img.shape[0])
    birdeye_warped = cv2.warpPerspective(img, trans_matrix, img_size)
    
    # Return the resulting image and matrix
    return birdeye_warped

prev_left_x = -1
prev_right_x = -1
prev_left_slope = None
prev_right_slope = None

def calculate_sliding_window(filtered_img):
    global prev_left_x
    global prev_right_x
    global prev_left_slope
    global prev_right_slope
    ##### Tunable parameter
    windows_num = 24
    window_width = 10
    # The part recognized from both edges is ignored
    outer_margin = 5
    inner_margin = 35
    consecutive_y_margin = 100
    # The x of succesive sliding windows should not differ by more than this value
    noise_threshold = 17
    # number of sliding window should larger than window_threshold
    window_threshold = 6
    # curve_threshold = 15
    slope_threshold = 20

    out_img = np.dstack((filtered_img, filtered_img, filtered_img))*255
    window_height = np.int(filtered_img.shape[0]/windows_num)
    consecutive_y_idx = consecutive_y_margin // window_height
    x_mid = np.int(filtered_img.shape[1]/2)

    # lw_arr : left sliding window array's position
    # rw_arr : right sliding window array's position
    lw_arr = []
    rw_arr = []
    for window_idx in range(windows_num):
        window_top = np.int(window_height * (windows_num - window_idx - 1))
        window_bottom = np.int(window_height * (windows_num - window_idx))

        leftx = find_firstfit(np.sum(filtered_img[window_top:window_bottom,:x_mid-inner_margin], axis=0), -1)
        rightx = find_firstfit(np.sum(filtered_img[window_top:window_bottom,x_mid+inner_margin:], axis=0), 1)+inner_margin
        
        if leftx > outer_margin and leftx < x_mid - inner_margin:
            if not lw_arr:
                # if prev_left_x == -1 or abs(prev_left_x - leftx) < noise_threshold:
                lw_arr.append((leftx, window_idx))
                cv2.rectangle(out_img,(leftx-window_width,window_bottom),(leftx,window_top),(0,255,0), 2)
            elif (window_idx - lw_arr[-1][1]) > consecutive_y_idx:
                pass
            elif abs(lw_arr[-1][0] - leftx) < noise_threshold:
                lw_arr.append((leftx, window_idx))
                cv2.rectangle(out_img,(leftx-window_width,window_bottom),(leftx,window_top),(0,255,0), 2)
        if rightx > inner_margin and rightx < x_mid - outer_margin:
            if not rw_arr:
                # if prev_right_x == -1 or abs(prev_right_x - (rightx+x_mid)) < noise_threshold:
                rw_arr.append((rightx + x_mid, window_idx))
                cv2.rectangle(out_img,(rightx + x_mid,window_bottom),(rightx + x_mid + window_width,window_top),(0,255,0), 2)
            elif (window_idx - rw_arr[-1][1]) > consecutive_y_idx:
                pass
            elif abs(rw_arr[-1][0] - (rightx + x_mid)) < noise_threshold:
                rw_arr.append((rightx + x_mid, window_idx))
                cv2.rectangle(out_img,(rightx + x_mid,window_bottom),(rightx + x_mid + window_width,window_top),(0,255,0), 2) 

    ### Fit a first order polynomial to each sliding windows
    isLeftValid = len(lw_arr) >= window_threshold
    isRightValid = len(rw_arr) >= window_threshold
    left_slope = -1
    right_slope = -1
    first_left_x_margin = 0
    first_right_x_margin = 0

    if isLeftValid:
        try:
            left_slope = math.degrees(math.atan(np.polyfit([x for (x, y) in lw_arr], [y * window_height for (x, y) in lw_arr], 1)[0]))
            if left_slope > 0:
                left_slope = 90 - left_slope
            elif left_slope < 0:
                left_slope = -90 - left_slope

            if left_slope < -slope_threshold:
                isLeftValid = False
                
        except:
            isLeftValid = False

    if isRightValid:
        try:
            right_slope = math.degrees(math.atan(np.polyfit([x for (x, y) in rw_arr], [y * window_height for (x, y) in rw_arr], 1)[0]))
            if right_slope > 0:
                right_slope = 90 - right_slope
            elif right_slope < 0:
                right_slope = -90 - right_slope

            if right_slope > slope_threshold:
                isRightValid = False
        except:
            isRightValid = False

    left_idx = 0
    right_idx = 0
    if isLeftValid and isRightValid:
        while left_idx < len(lw_arr) and right_idx < len(rw_arr) and lw_arr[left_idx][1] != rw_arr[right_idx][1]:
            if lw_arr[left_idx][1] > rw_arr[right_idx][1]:
                right_idx += 1
            else:
                left_idx += 1
        if left_idx < len(lw_arr) and right_idx < len(rw_arr):
            first_left_x_margin = lw_arr[left_idx][0]
            first_right_x_margin = filtered_img.shape[1] - rw_arr[right_idx][0]
    elif isLeftValid:
        first_left_x_margin = lw_arr[0][0]
    elif isRightValid:
        first_right_x_margin = filtered_img.shape[1] - rw_arr[0][0]
    
    if left_idx == len(lw_arr):
        isLeftValid = False
    if right_idx == len(rw_arr):
        isRightValid = False
    
    if isLeftValid:
        prev_left_x = lw_arr[0][0]
        prev_left_slope = left_slope
    if isRightValid:
        prev_right_x = rw_arr[0][0]
        prev_right_slope = right_slope

    return out_img, left_slope, right_slope, \
        isLeftValid, isRightValid, first_left_x_margin, first_right_x_margin

class lane_keeping_module:
    def __init__(self, config_dict):
        self.twist_pub = rospy.Publisher('twist_cmd', TwistStamped, queue_size = 10)
        self.filter_thr_dict = config_dict['filter_thr_dict']
        self.birdeye_warp_param = config_dict['birdeye_warp_param']

        self.velocity = config_dict['velocity']
        self.steer_sensitivity = config_dict['steer_sensitivity']
        self.debug_window = config_dict['debug_window']

        self.image_width = 320
        self.image_height = 240
        self.turn_state = turnState.FORWARD

        if self.debug_window:
            self.trackbar_img = np.zeros((1,400), np.uint8)
            # Create Window
            cv2.namedWindow('original_image')
            cv2.namedWindow('birdeye_image')
            cv2.namedWindow('sliding_window')
            cv2.namedWindow('filtered_birdeye')
            cv2.namedWindow('TrackBar')

            # Move Window Location              #col    #row
            cv2.moveWindow('original_image',    450*0,  350*0)
            cv2.moveWindow('birdeye_image',     450*0,  350*1)
            cv2.moveWindow('sliding_window',    450*1,  350*0)
            cv2.moveWindow('filtered_birdeye',  450*1,  350*1)
            cv2.moveWindow('TrackBar',          450*2,  350*0)

            # Create Trackbar
            cv2.createTrackbar("[clr]x_grad_min", "TrackBar", self.filter_thr_dict['x_grad_thr'][0], 255, self.onChange)
            cv2.createTrackbar("[clr]x_grad_max", "TrackBar", self.filter_thr_dict['x_grad_thr'][1], 255, self.onChange)
            cv2.createTrackbar("[clr]sat_min", "TrackBar", self.filter_thr_dict['saturation_thr'][0], 255, self.onChange)
            cv2.createTrackbar("[clr]sat_max", "TrackBar", self.filter_thr_dict['saturation_thr'][1], 255, self.onChange)
            cv2.createTrackbar("[clr]red_min", "TrackBar", self.filter_thr_dict['red_thr'][0], 255, self.onChange)
            cv2.createTrackbar("[clr]red_max", "TrackBar", self.filter_thr_dict['red_thr'][1], 255, self.onChange)
            cv2.createTrackbar("[clr]green_min", "TrackBar", self.filter_thr_dict['green_thr'][0], 255, self.onChange)
            cv2.createTrackbar("[clr]green_max", "TrackBar", self.filter_thr_dict['green_thr'][1], 255, self.onChange)
            cv2.createTrackbar("[clr]blue_min", "TrackBar", self.filter_thr_dict['blue_thr'][0], 255, self.onChange)
            cv2.createTrackbar("[clr]blue_max", "TrackBar", self.filter_thr_dict['blue_thr'][1], 255, self.onChange)
            cv2.createTrackbar("[be]top", "TrackBar", self.birdeye_warp_param['top'], 240, self.onChange)
            cv2.createTrackbar("[be]bottom", "TrackBar", self.birdeye_warp_param['bottom'], 240, self.onChange)
            cv2.createTrackbar("[be]top_width", "TrackBar", self.birdeye_warp_param['top_width'], 320, self.onChange)
            cv2.createTrackbar("[be]bottom_width", "TrackBar", self.birdeye_warp_param['bottom_width'], 320, self.onChange)
            cv2.createTrackbar("[be]birdeye_width", "TrackBar", self.birdeye_warp_param['birdeye_width'], 320, self.onChange)

    def onChange(self, pos):
        pass

    def config_image_source(self, mode='webcam'):
        if mode == 'webcam':
            # VideoCapture(n) : n th input device (PC : 0, minicar : 1)
            self.capture = cv2.VideoCapture(1)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
        elif mode == 'video':
            video_file = './test_video.avi'
            self.capture = cv2.VideoCapture(video_file)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)

    def calculate_velocity_and_angle(self, ls, rs, lv, rv, lx, rx):
        # Tunable parameter
        left_ground_truth = 4
        right_ground_truth = -4
        left_steer = 13
        left_steer_threshold = 2
        right_steer = -13
        right_steer_threshold = -2

        velocity = self.velocity
        target_angle = 0

        if self.turn_state == turnState.LEFT:
            if (lv and ls > 0):
                self.turn_state = turnState.FORWARD
            if rv:
                target_angle = max((rx - self.image_height / 6)/3, left_steer_threshold)
            else:
                target_angle = left_steer
        elif self.turn_state == turnState.RIGHT:
            if (rv and rs < 0):
                self.turn_state = turnState.FORWARD
            if lv:
                target_angle = min((self.image_height / 6 - lx)/3, right_steer_threshold)
            else:
                target_angle = right_steer
        else:
            # If two lines are valid, determine by first x position
            if lv and rv:
                if ls < right_ground_truth and rs < right_ground_truth * 3:
                    self.turn_state = turnState.LEFT
                if rs > left_ground_truth and ls > left_ground_truth * 3:
                    self.turn_state = turnState.RIGHT
                target_angle = (rx - lx) / 15
            elif rv:
                target_angle = (rx - self.image_height / 8) / 10
                if rs < right_ground_truth * 3:
                    self.turn_state = turnState.LEFT
                if rs > left_ground_truth:
                    self.turn_state = turnState.RIGHT
            elif lv:
                target_angle = (self.image_height / 8 - lx) / 10
                if ls < right_ground_truth:
                    self.turn_state = turnState.LEFT
                if ls > left_ground_truth * 3:
                    self.turn_state = turnState.RIGHT

        if self.debug_window:
            if lv:
                print('left :', ls, lx)
            if rv:
                print('right :', rs, rx)

        target_angle = target_angle * self.steer_sensitivity

        # positive for left turn        
        return velocity, target_angle

    def svl_spinner(self):
        self.image_sub = rospy.Subscriber('/simulator/camera_node/image/compressed', CompressedImage, self.image_callback)
        rospy.spin()

    def image_callback(self, data):
        np_arr = np.fromstring(data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if self.debug_window:
            self.filter_thr_dict['x_grad_thr'][0] = cv2.getTrackbarPos("[clr]x_grad_min", "TrackBar")
            self.filter_thr_dict['x_grad_thr'][1] = cv2.getTrackbarPos("[clr]x_grad_max", "TrackBar")
            self.filter_thr_dict['saturation_thr'][0] = cv2.getTrackbarPos("[clr]sat_min", "TrackBar")
            self.filter_thr_dict['saturation_thr'][1] = cv2.getTrackbarPos("[clr]sat_max", "TrackBar")
            self.filter_thr_dict['red_thr'][0] = cv2.getTrackbarPos("[clr]red_min", "TrackBar")
            self.filter_thr_dict['red_thr'][1] = cv2.getTrackbarPos("[clr]red_max", "TrackBar")
            self.filter_thr_dict['green_thr'][0] = cv2.getTrackbarPos("[clr]green_min", "TrackBar")
            self.filter_thr_dict['green_thr'][1] = cv2.getTrackbarPos("[clr]green_max", "TrackBar")
            self.filter_thr_dict['blue_thr'][0] = cv2.getTrackbarPos("[clr]blue_min", "TrackBar")
            self.filter_thr_dict['blue_thr'][1] = cv2.getTrackbarPos("[clr]blue_max", "TrackBar")
            self.birdeye_warp_param['top'] = cv2.getTrackbarPos("[be]top", "TrackBar")
            self.birdeye_warp_param['bottom'] = cv2.getTrackbarPos("[be]bottom", "TrackBar")
            self.birdeye_warp_param['top_width'] = cv2.getTrackbarPos("[be]top_width", "TrackBar")
            self.birdeye_warp_param['bottom_width'] = cv2.getTrackbarPos("[be]bottom_width", "TrackBar")
            self.birdeye_warp_param['birdeye_width'] = cv2.getTrackbarPos("[be]birdeye_width", "TrackBar")

        cv2.waitKey(1)

        birdeye_image = birdeye_warp(image_np, self.birdeye_warp_param)
        filtered_birdeye = color_gradient_filter(birdeye_image, self.filter_thr_dict)
        sliding_window, ls, rs, lv, rv, lx, rx = calculate_sliding_window(filtered_birdeye)

        if self.debug_window:
            cv2.imshow('TrackBar', self.trackbar_img)
            cv2.imshow('original_image', image_np)
            cv2.imshow('birdeye_image', birdeye_image)
            cv2.imshow('sliding_window', sliding_window)
            cv2.imshow('filtered_birdeye', (filtered_birdeye*255).astype(np.uint8))

        msg = TwistStamped()
        velocity, angle = self.calculate_velocity_and_angle(ls, rs, lv, rv, lx, rx)

        if self.debug_window:
            print('-------------------------------')
            print('State : ', self.turn_state)
            print('Angle : ', round(angle, 3))

        msg.twist.linear.x = velocity
        msg.twist.angular.z = angle
        self.twist_pub.publish(msg)

    def twist_publisher(self):
        rate = rospy.Rate(10) # 10hz
        while not rospy.is_shutdown():
            _, original_image = self.capture.read()

            if self.debug_window:
                self.filter_thr_dict['saturation_thr'][0] = cv2.getTrackbarPos("[clr]sat_min", "TrackBar")
                self.filter_thr_dict['saturation_thr'][1] = cv2.getTrackbarPos("[clr]sat_max", "TrackBar")
                self.filter_thr_dict['red_thr'][0] = cv2.getTrackbarPos("[clr]red_min", "TrackBar")
                self.filter_thr_dict['red_thr'][1] = cv2.getTrackbarPos("[clr]red_max", "TrackBar")
                self.filter_thr_dict['green_thr'][0] = cv2.getTrackbarPos("[clr]green_min", "TrackBar")
                self.filter_thr_dict['green_thr'][1] = cv2.getTrackbarPos("[clr]green_max", "TrackBar")
                self.filter_thr_dict['blue_thr'][0] = cv2.getTrackbarPos("[clr]blue_min", "TrackBar")
                self.filter_thr_dict['blue_thr'][1] = cv2.getTrackbarPos("[clr]blue_max", "TrackBar")
                self.birdeye_warp_param['top'] = cv2.getTrackbarPos("[be]top", "TrackBar")
                self.birdeye_warp_param['bottom'] = cv2.getTrackbarPos("[be]bottom", "TrackBar")
                self.birdeye_warp_param['top_width'] = cv2.getTrackbarPos("[be]top_width", "TrackBar")
                self.birdeye_warp_param['bottom_width'] = cv2.getTrackbarPos("[be]bottom_width", "TrackBar")
                self.birdeye_warp_param['birdeye_width'] = cv2.getTrackbarPos("[be]birdeye_width", "TrackBar")

            cv2.waitKey(1)

            birdeye_image = birdeye_warp(original_image, self.birdeye_warp_param)
            filtered_birdeye = color_gradient_filter(birdeye_image, self.filter_thr_dict)
            sliding_window, ls, rs, lv, rv, lx, rx = calculate_sliding_window(filtered_birdeye)

            if self.debug_window:
                cv2.imshow('original_image', original_image)
                cv2.imshow('birdeye_image', birdeye_image)
                cv2.imshow('sliding_window', sliding_window)
                cv2.imshow('filtered_birdeye', (filtered_birdeye*255).astype(np.uint8))
                cv2.imshow('TrackBar', self.trackbar_img)

            msg = TwistStamped()
            velocity, angle = self.calculate_velocity_and_angle(ls, rs, lv, rv, lx, rx)

            if self.debug_window:
                print('-------------------------------')
                print('State : ', self.turn_state)
                print('Angle : ', round(angle, 3))

            msg.twist.linear.x = velocity
            msg.twist.angular.z = angle
            self.twist_pub.publish(msg)
            rate.sleep()
        
        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        with open('config/config') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
    except:
        print('Should make config file in config folder!')
        exit(1)

    rospy.init_node('lkas')

    # Mode : webcam, svl, video
    ic = lane_keeping_module(config_dict)
    if config_dict['mode'] == 'svl':
        ic.svl_spinner()
    elif config_dict['mode'] in ['webcam', 'video']:
        ic.config_image_source(config_dict['mode'])
        ic.twist_publisher()
    else:
        print('Should select appropriate mode! (webcam, svl, video)')
        exit(1)

