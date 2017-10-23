# -*- coding: utf-8 -*-

def covert_to_height(input_number, height):
    return int(((input_number) / 480.0) * height)

def covert_to_width(input_number, width):
    return int(((input_number) / 640.0) * width)