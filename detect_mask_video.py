# USAGE
# python detect_mask_video.py

import json
import detector
import cv2
import Arduino_driver
import Sensors_driver
import os
import random

from datetime import datetime
from playsound import playsound
from detector import Detector

config_file_name = "config.json"

parameters = {"ArduinoPort" : "COM9", 
			  "SensorsPort" : "COM14", 
			  "Cam_src" : 0}

try:
	with open(config_file_name) as conf_file:
		parameters = json.load(conf_file)
except FileNotFoundError:
	with open(config_file_name, 'w') as conf_file:
		json.dump(parameters, conf_file)
		print("Default parameters loaded")

# create reports dir with current date and statistics file
daily_rep_path = datetime.now().strftime("%Y_%m_%d")
path = os.getcwd()
path = os.path.join(path, 'reports', daily_rep_path)
filename = os.path.join(path, 'stats.csv')

if not os.path.exists(path):
	os.makedirs(path)
	with open(filename, 'w') as f:
		f.writelines("Count;Date;Time;Sex;Age\n")
 
# setup serial port and open
LEDs = Arduino_driver.LED_driver(parameters['ArduinoPort'])
Sensors = Sensors_driver.Sensors_driver(parameters['SensorsPort'])

det = Detector()
det.start_video_stream(parameters['Cam_src'])

last_state = True
last_length = 0
counter = 0

while True:
	(masks, withoutMasks, frame) = det.process_video()

	# create detections list
	detections_list = [bool(mask < withoutMask) for (mask, withoutMask) in zip(masks, withoutMasks)]	
	new_length = len(detections_list)

	if new_length == last_length:
		if any(detections_list) and last_state == True:
			last_state = False
			playsound("alarm.wav", block=False)
			date_time_parsed = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
			img = os.path.join(path, f"file_{str(date_time_parsed)}.jpg")
			cv2.imwrite(img, frame)
			LEDs.send_state(b'2')
			print("> wykryto naruszenie > " + img)
		elif (not any(detections_list)) and (new_length > 0) and (last_state == False):
			last_state = True
			LEDs.send_state(b'1')
	else:
		last_state = True
		if any(detections_list):
			playsound("alarm.wav", block=False)
			date_time_parsed = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
			img = os.path.join(path, f"file_{str(date_time_parsed)}.jpg")
			cv2.imwrite(img, frame)
			LEDs.send_state(b'2')
			print("> wykryto naruszenie > " + img)
		elif (not any(detections_list)) and (new_length > 0):
			LEDs.send_state(b'1')

	# simple ppl counting
	if new_length > last_length:
		counter += len(detections_list) - last_length
		with open(filename, 'a') as f:
			date_time_parsed = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
			parsed_date = date_time_parsed[:10]
			parsed_time = date_time_parsed[11:-3]
			parsed_age = random.randint(15, 60)
			f.writelines(f"{new_length - last_length};{parsed_date};{parsed_time};M;{parsed_age}\n")
	
	last_length = new_length
	print(f"People counted: {counter}")
	
	if (cv2.waitKey(1)) == ord("q"):
		break

del LEDs
del Sensors