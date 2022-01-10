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

BUZZER_GPIO = 22


def make_push_notification():
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


def play_tone():
    player = TonePlayer(gpio=BUZZER_GPIO, bpm=10)
    player.play("E6q")
    player.play("C6q")


def use_leds():
    with Leds() as leds:
        leds.update(Leds.rgb_on(Color.RED))
        time.sleep(1)
        leds.update(Leds.rgb_off())
        time.sleep(1)


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

    with PiCamera() as camera:
        # Configure camera
        camera.resolution = (410, 230)  # Full Frame, 16:9 (Camera v2)
        # camera.start_preview()

        # Do inference on VisionBonnet
        with CameraInference(object_detection.model()) as inference:
            for result in inference.run():
                objects = object_detection.get_objects(result, args.threshold)

                for obj in objects:
                    print("Object was detected: %s" % (obj))
                    if obj.kind in [2, 3]:
                        camera.capture("image.jpg")
                        make_push_notification()
                        play_tone()
                        use_leds()


if __name__ == "__main__":
    main()
