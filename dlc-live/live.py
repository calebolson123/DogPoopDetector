from dlclive import DLCLive, Processor

import math
from collections import deque
import os
import cv2
import numpy as np
import time
from csv import writer, DictReader
from reolinkapi import Camera # or hopefully another existing API for interfacing w/ your video feed/device
from math import atan2, pi

# TODO: in general this is big and could be broken into separate files
class PoopDetector():

    ''' v3
    - 0 spine1
    - 1 spine2
    - 2 spine3
    - 3 tailstart
    - 4 tailmid
    - 5 tailend
    - 6 butt
    '''

    def __init__(self, cast_service):
        self.cast_service = cast_service
        self.fps = 30
        self.next_frame = 0
        self.pooping_threshold = 0.65
        self.last_spine_location = None
        self.moving = False
        self.movement_timestamp = time.time()
        self.movement_threshold_pixels = 25
        self.seconds_to_determine_movement = 2
        self.pooping_timestamp = time.time()
        self.seconds_to_determine_pooping = 3

        # A rolling capped queue is used for resiliency against individual frames that look like pooping
        self.pooping_detected_queue = deque(maxlen=self.fps*self.seconds_to_determine_pooping)

         # update to your resolution, lower the better for performance-sake.
        self.frame_dim = (1280,720)
        self.dlc_live = DLCLive(
            '../exported-models/DLC_SecurityCamPooping_resnet_50_iteration-0_shuffle-1',
            display=True,
            display_radius=1.5,
            display_cmap='glasbey_light'
        )


    def slope(self, x1, y1, x2, y2):
        return (y2-y1) / (x2-x1)


    # angle in degrees between line through a -> b -> c,wrapping around b
    def angle(self, a, b, c):
        ang = math.degrees(
            math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        return ang + 360 if ang < 0 else ang


    # heuristics for poop tail, returns bool depending on whether poop tail found
    def check_for_poop_tail(self, pose):
        tail_detected = False if any(e < 0.8 for (e) in pose[3:6, 2]) else True
        if not tail_detected:
            return False

        # TODO: ew, cleanup & decouple hardcodings to original device
        # TODO: remove ambiguous index access
        if abs(self.angle([pose[3,0],pose[3,1]], [pose[4,0],pose[4,1]], [pose[5,0],pose[5,1]]) - 180) < 40:
            # if tail is straight up in Y direction (i.e. up while butt facing camera)
            # check whether it is in a straight line w/ the butt location... if so, dont return True
            if pose[6,2] > 0.8:
                # if tail end is within N points in either direction on X axis
                # if in top section of frame (way far back, less than 125px from top) dont cancel out
                if abs(pose[5,0] - pose[6,0]) < 5 and pose[6,1] > 125:
                    return False
            return True
        return False


    # simple heuristic checking for rounded spine, i.e. hunchback poop position lol
    def check_for_poop_spine(self, pose):
        spine_detected = False if any(e < 0.8 for (e) in pose[0:3, 2]) else True
        if not spine_detected:
            return False

        spine_section_1_slope = self.slope(pose[0,0],pose[0,1],pose[1,0],pose[1,1])
        spine_section_2_slope = self.slope(pose[1,0],pose[1,1],pose[2,0],pose[2,1])

        if abs(spine_section_1_slope - spine_section_2_slope) > 0.35: # this value may be different for your dog
            return True

        return False


    # return bool for whether the dog "is moving" or appears to be still
    # movement detection is based on spine location. If spine isn't present,
    # return False, i.e. not moving or just not present in frame
    def check_for_movement(self, pose):

        # for performance, just check for movement every N seconds
        should_check_movement = time.time() >= self.movement_timestamp + self.seconds_to_determine_movement

        if should_check_movement:
            self.movement_timestamp = time.time()
            spine_detected = False if pose[1, 2] < 0.8 else True
            if not spine_detected:
                self.moving = False
                return self.moving

            current_spine_location = (pose[1,0], pose[1,1])

            if self.last_spine_location is not None:
                x_diff = abs(current_spine_location[0] - self.last_spine_location[0])
                y_diff = abs(current_spine_location[1] - self.last_spine_location[1])

                if x_diff > self.movement_threshold_pixels or y_diff > self.movement_threshold_pixels:
                    self.moving = True
                else:
                    self.moving = False

            self.last_spine_location = current_spine_location
    
        return self.moving
        

    # leverage all the pooping heuristics to determine
    # whether pooping is detected in this frame
    def is_twinkie_pooping(self, pose):

        # 1) look for poop spine
        poop_spine = self.check_for_poop_spine(pose)

        # 2) look for pointy poop tail
        poop_tail = self.check_for_poop_tail(pose)

        # 3) not moving "for awhile"
        moving = self.check_for_movement(pose)

        # print([poop_spine, poop_tail, not moving])
        pooping = all([poop_spine, poop_tail, not moving])
        if pooping:
            return 1

        return 0


    def check_rolling_average(self, pose, img):
        # check rolling average
        if sum(self.pooping_detected_queue) / len(self.pooping_detected_queue) > self.pooping_threshold:
            print('\nPooping detected')
            # self.cast_service.play_sound()
            self.save_poop_location(pose,img)
            # clear once poop detected, helps w/ trailing additional poo detections due to filled queue
            self.pooping_detected_queue.clear() 


    # This is great for debugging/testing with recorded video
    def beefy_boy(self):
        #####

        # For using with live camera, you'll need to replace the recorded images
        # with images streamed via RTSP (probably). Depending on your camera,
        # you'll need to find an API for connecting to it.

        #####
        cap = cv2.VideoCapture('path_to_sample_video.mp4')
        success, img = cap.read()
        fno = 0
        self.dlc_live.init_inference(self.maintain_aspect_ratio_resize(img, width=self.frame_dim[0], height=self.frame_dim[1])) # for some reason, gotta prime the pump
        while success:
            
            # this gnarliness is to throttle the video to process at your set fps
            frame = None
            while frame is None:
                cur_time = time.time()
                if cur_time > self.next_frame:
                    frame = img
                    self.next_frame = max(
                        self.next_frame + 1.0 / self.fps, cur_time + 0.5 / self.fps
                    )

            success, img = cap.read()

            if all(e is not None for e in [frame, img]):
                pose = self.dlc_live.get_pose(self.maintain_aspect_ratio_resize(img, width=self.frame_dim[0], height=self.frame_dim[1]))
                is_pooping = self.is_twinkie_pooping(pose)
                self.pooping_detected_queue.append(is_pooping)

                time_to_check_average = time.time() >= self.pooping_timestamp + self.seconds_to_determine_pooping
                if time_to_check_average:
                    self.pooping_timestamp = time.time()
                    print('\npooping likelihood: ', sum(self.pooping_detected_queue) / len(self.pooping_detected_queue), '\n')
                    self.check_rolling_average(pose, img)


    # This is should be broken up, but this does the following:
    # 1) store poop locations (x,y) in csv
    # 2) draw circle on "this" frame, beneath the base of the tail
    # 3) saves the image so you can access the picture from any device on LAN
    def save_poop_location(self, pose, img):
        # attempt to save under tail_start
        x, y = pose[3,0], pose[3,1] + 15
        poop_location = [x,y]

        write = True
        with open('poop_locations.csv') as f:
            reader = DictReader(f)
            for row in reader:
                # draw poop on image
                cv2.circle(img, (int(float(row['x'])), int(float(row['y']))), 15, (0,0,255), thickness=2, lineType=8)

                # dont write to csv if already detected poop
                distance = math.sqrt(((x - float(row['x'])) ** 2) + ((y - float(row['y'])) ** 2))
                if distance < 50:
                    write = False

            f.close()

        # save image
        if write: cv2.circle(img, (int(poop_location[0]), int(poop_location[1])), 15, (0,0,255), thickness=2, lineType=8)
        cv2.imwrite('poop_locations.jpg', img)
                
        if write:
            with open('poop_locations.csv', 'a', newline='', encoding='utf-8') as f:
                writer_object = writer(f)
                writer_object.writerow(poop_location)
                f.close()


    # Resizes a image and maintains aspect ratio
    def maintain_aspect_ratio_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        # Grab the image size and initialize dimensions
        dim = None
        (h, w) = image.shape[:2]

        # Return original image if no need to resize
        if width is None and height is None:
            return image

        # We are resizing height if width is none
        if width is None:
            # Calculate the ratio of the height and construct the dimensions
            r = height / float(h)
            dim = (int(w * r), height)
        # We are resizing width if height is none
        else:
            # Calculate the ratio of the 0idth and construct the dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # Return the resized image
        return cv2.resize(image, dim, interpolation=inter)