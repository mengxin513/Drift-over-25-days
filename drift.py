from openflexure_microscope import load_microscope
from contextlib import closing
import data_file
from openflexure_microscope.microscope import picamera_supports_lens_shading
import busio
import board
import adafruit_sht31d
import numpy as np
import cv2
from PIL import Image
from camera_stuff import find_template
import queue
import threading
import time

def image_capture(start_t, event, ms, q, sensor):
    while event.is_set():
        frame = ms.rgb_image().astype(np.float32)
        capture_t = time.time()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        q.put(frame)
        tim = capture_t - start_t
        q.put(tim)
        q.put(sensor.relative_humidity)
        q.put(sensor.temperature)
        print('Number of itms in the queue: {}'.format(q.qsize()))

if __name__ == "__main__":

    with load_microscope("microscope_settings.npz") as ms, \
         closing(data_file.Datafile(filename = "drift.hdf5")) as df:

        assert picamera_supports_lens_shading(), "You need the updated picamera module with lens shading!"

        camera = ms.camera
        stage = ms.stage

        i2c = busio.I2C(board.SCL, board.SDA)
        sensor = adafruit_sht31d.SHT31D(i2c)

        camera.resolution = (640, 480)

        drift_data = df.new_group("data", "time,humidity,temperature,camerax,cameray")

        N_frames = 1000
        #need to be consistant between drift.py and drift_plot.py

        camera.start_preview(resolution = (640, 480))

        image = ms.rgb_image().astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean = np.mean(image)
        templ8 = (image - mean)[144:-144, 144:-144]
        drift_data['template'] = templ8
        drift_data['initial_image'] = image
        imgfile_location = "/home/pi/drift/calibration/drift_templ8.jpg"
        cv2.imwrite(imgfile_location, templ8)
        imgfile_location = "/home/pi/drift/calibration/drift_image.jpg"
        cv2.imwrite(imgfile_location, image)
        img = Image.open("/home/pi/drift/calibration/drift_templ8.jpg")
        pad = Image.new('RGB', (352, 192)) #Tuple must be multiples of 32 and 16
        pad.paste(img, (0, 0))
        overlay = camera.add_overlay(pad.tobytes(), size = (352, 192))
        overlay.alpha = 128
        overlay.fullscreen = False
        overlay.layer = 3
        
        templ8_position = np.zeros((1, 2))
        frame = ms.rgb_image().astype(np.float32)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        templ8_position[0, :], corr = find_template(templ8, frame - np.mean(frame), return_corr = True, fraction = 0.3)

        def move_overlay(cx, cy):
            """move the overlay to show a shift of cx,cy camera pixels"""
            x = int(400 + (cx - templ8_position[0, 0] - 176) * 1)
            y = int(240 + (cy - templ8_position[0, 1] - 96) * 1)
            overlay.window = (x, y, int(352 * 1), int(192 * 1))

        q = queue.Queue()
        event = threading.Event()

        start_t = time.time()
        t = threading.Thread(target = image_capture, args = (start_t, event, ms, q, sensor), name = 'thread1')
        event.set()
        t.start()

        try:
            while event.is_set():
                if not q.empty():
                    data = np.zeros((N_frames, 5))
                    for i in range(N_frames):
                        frame = q.get()
                        tim = q.get()
                        data[i, 0] = tim
                        data[i, 1] = q.get() #relative humidity 0-100%
                        data[i, 2] = q.get() #temperature measured in degrees Celsius
                        data[i, 3:], corr = find_template(templ8, frame - np.mean(frame), return_corr = True, fraction = 0.3)
                        move_overlay(*data[i, 3:5])
                        camera.stop_preview()
                        cv2.imshow("corr", corr / np.max(corr))
                        cv2.waitKey(500)
                        camera.start_preview(resolution = (640, 480))
                    df.add_data(data, drift_data, "data")
                    imgfile_location_1 = "/home/pi/drift/frames/drift_%s.jpg" % time.strftime("%Y%m%d_%H%M%S")
                    imgfile_location_2 = "/home/pi/drift/frames/corr_%s.jpg" % time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(imgfile_location_1, frame)
                    cv2.imwrite(imgfile_location_2, corr * 255.0 / np.max(corr))
                else:
                    time.sleep(0.5)
                print("Looping")
            print("Done")
        except KeyboardInterrupt:
            event.clear()
            t.join()
            camera.stop_preview()
            print ("Got a keyboard interrupt, stopping")
            