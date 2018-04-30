#!/bin/bash

ffmpeg -r 5 -i ../implementations/acgan/images/%d.png acgan.gif
