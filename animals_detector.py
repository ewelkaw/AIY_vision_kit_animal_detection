#!/usr/bin/env python3
import argparse
import os
import time

import requests
from aiy.leds import Color, Leds
from aiy.toneplayer import TonePlayer
from aiy.vision.inference import CameraInference
from aiy.vision.models import object_detection
from picamera import PiCamera


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.7,
        help="Detection probability threshold.",
    )
    args = parser.parse_args()
    BUZZER_GPIO = 22
    player = TonePlayer(gpio=BUZZER_GPIO, bpm=10)

    with PiCamera() as camera:
        # Configure camera
        camera.resolution = (410, 230)  # Full Frame, 16:9 (Camera v2)
        # camera.start_preview()

        # Do inference on VisionBonnet
        with CameraInference(object_detection.model()) as inference:
            for result in inference.run():

                objects = object_detection.get_objects(result, args.threshold)

                for i, obj in enumerate(objects):
                    print("Object #%d: %s" % (i, obj))
                    if obj.kind in [2, 3]:
                        camera.capture("image.jpg")
                        requests.post(
                            "https://api.pushover.net/1/messages.json",
                            data={
                                "token": os.environ("TOKEN"),
                                "user": os.environ("USER"),
                                "message": "Animal detected",
                            },
                            files={
                                "attachment": (
                                    "image.jpg",
                                    open("image.jpg", "rb"),
                                    "image/jpeg",
                                )
                            },
                        )
                        player.play("E6q")
                        player.play("C6q")
                        with Leds() as leds:
                            leds.update(Leds.rgb_on(Color.RED))
                            time.sleep(1)
                            leds.update(Leds.rgb_off())
                            time.sleep(1)


if __name__ == "__main__":
    main()
