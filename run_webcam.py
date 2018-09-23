# © 2018, modified to send the detect body parts over OSC
# Mosalam Ebrahimi and Lilac Atassi

import argparse
import logging
import time

import cv2

from pythonosc import udp_client

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


def send_message(name, number):
    if number in humans[0].body_parts.keys():
        body_part = humans[0].body_parts[number]
        client.send_message(name, [body_part.x, body_part.y])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')

    parser.add_argument('--ip', type=str, default='127.0.0.1',
                        help='the ip address of the OSC server. default=127.0.0.1')
    parser.add_argument('--port', type=int, default=57120,
                        help='the port number of the OSC server. default=57120 supercollider')

    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    client = udp_client.SimpleUDPClient(args.ip, args.port)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(time.time())+'.avi', fourcc, 20.0, (image.shape[1], image.shape[0]))

    while True:
        ret_val, image = cam.read()

        # out.write(image)

        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

        if len(humans) > 0:
            body_parts = [("/right_wrist", 4), ("/left_wrist", 7),
                          ("/right_elbow", 3), ("/left_elbow", 6),
                          ("/right_shoulder", 2), ("/left_shoulder", 5),
                          ("/right_knee", 9), ("/left_knee", 12),
                          ("/right_ankle", 10), ("/left_ankle", 13),
                          ("/right_hip", 8), ("/left_hip", 11)]
            for body_part in body_parts:
                send_message(body_part[0], body_part[1])

    out.release()
    cv2.destroyAllWindows()
