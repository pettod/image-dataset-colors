import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from tqdm import tqdm


IMAGE_PATHS = glob("flowers.jpeg")
DOWNSAMPLING_FACTOR = 8
COLORS_IN_GROUP = 13
CSV_FILE_NAME = "hex_colors.csv"


def rgb2lab(inputColor):

   num = 0
   RGB = [0, 0, 0]

   for value in inputColor :
       value = float(value) / 255

       if value > 0.04045 :
           value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
       else :
           value = value / 12.92

       RGB[num] = value * 100
       num = num + 1

   XYZ = [0, 0, 0,]

   X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
   Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
   Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
   XYZ[ 0 ] = round( X, 4 )
   XYZ[ 1 ] = round( Y, 4 )
   XYZ[ 2 ] = round( Z, 4 )

   XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
   XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
   XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

   num = 0
   for value in XYZ :

       if value > 0.008856 :
           value = value ** ( 0.3333333333333333 )
       else :
           value = ( 7.787 * value ) + ( 16 / 116 )

       XYZ[num] = value
       num = num + 1

   Lab = [0, 0, 0]

   L = ( 116 * XYZ[ 1 ] ) - 16
   a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
   b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

   Lab [ 0 ] = round( L, 4 )
   Lab [ 1 ] = round( a, 4 )
   Lab [ 2 ] = round( b, 4 )

   return Lab


class Color():
    def __init__(self, color_file_name):
        # Read colors to gropus
        self.colors = pd.read_csv(color_file_name)
        self.number_groups = len(self.colors) // COLORS_IN_GROUP
        self.color_groups = [[] for i in range(self.number_groups)]
        self.color_brightnesses = [[] for i in range(self.number_groups)]
        self.image_rgb_colors = {}
        for i, c in enumerate(self.colors["hex"]):
            rgb = self.hexToRgb(c)
            lab = rgb2lab(rgb)
            self.image_rgb_colors[c] = [0, rgb, lab]
            self.color_groups[i // COLORS_IN_GROUP].append(c)
            self.color_brightnesses[i // COLORS_IN_GROUP].append(lab[0])
        self.color_brightnesses = np.mean(np.array(self.color_brightnesses), axis=1)

    def hexToRgb(self, hex_string):
        r_hex = hex_string[1:3]
        g_hex = hex_string[3:5]
        b_hex = hex_string[5:7]
        return [int(r_hex, 16), int(g_hex, 16), int(b_hex, 16)]

    def euclidianDistance(self, a, b):
        d = 0
        for i in range(len(a)):
            d += (a[i] - b[i]) ** 2
        return d

    def angleDistance(self, a, b):
        def angles(c):
            def angle(c1, c2):
                return (c1 + 1) / (c2 + 1)
            return [
                angle(c[0], c[1]),
                angle(c[1], c[2]),
                angle(c[2], c[0]),
            ]
        a_angles = angles(a)
        b_angles = angles(b)
        d = 0
        for i in range(len(a_angles)):
            d += (a_angles[i] - b_angles[i]) ** 2
        return d

    def fastColorFind(self, pixel):
        pixel_brightness = pixel[0]
        brightness_i = 0
        min_distance = abs(pixel_brightness - self.color_brightnesses[brightness_i])
        for i in range(1, len(self.color_brightnesses)):
            distance = abs(pixel_brightness - self.color_brightnesses[i])
            if distance < min_distance:
                min_distance = distance
                brightness_i = i
        #brightness_i = int(np.mean(pixel) / 255 * (COLORS_IN_GROUP - 1))
        min_group = 0
        min_hex = self.color_groups[min_group][brightness_i]
        min_distance = self.euclidianDistance(pixel, rgb2lab(self.image_rgb_colors[min_hex][2]))
        for i in range(1, len(self.color_groups)):
            hex = self.color_groups[i][brightness_i]
            distance = self.euclidianDistance(pixel, rgb2lab(self.image_rgb_colors[hex][2]))
            if distance < min_distance:
                min_distance = distance
                min_hex = hex
                min_group = i
        self.image_rgb_colors[min_hex][0] += 1

    def accurateColorFind(self, pixel):
        min_hex = self.color_groups[0][0]
        min_distance = self.angleDistance(pixel, self.image_rgb_colors[min_hex][0])
        for i in range(len(self.color_groups)):
            for j in range(len(self.color_groups[0])):
                hex = self.color_groups[i][j]
                distance = self.angleDistance(pixel, self.image_rgb_colors[hex][0])
                if distance < min_distance:
                    min_distance = distance
                    min_hex = hex
        self.image_rgb_colors[min_hex][1] += 1

    def plot(self):
        color_distribution = [[] for i in range(self.number_groups)]
        for i in range(len(self.color_groups)):
            for j in range(len(self.color_groups[0])):
                color_distribution[i].append(
                    self.image_rgb_colors[self.color_groups[i][j]][0])
        fig = plt.figure(figsize=(12,12), frameon=True)
        ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
        plt.table(
            color_distribution,
            cellColours=self.color_groups,
            loc="center",
            rowLabels=None,
            colWidths=None,
        )
        plt.savefig("color_table.png")


def main():
    color = Color(CSV_FILE_NAME)
    for image_path in IMAGE_PATHS:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        if DOWNSAMPLING_FACTOR > 1:
            h, w, c = image.shape
            image = cv2.resize(
                image,
                (h // DOWNSAMPLING_FACTOR, w // DOWNSAMPLING_FACTOR),
                interpolation=cv2.INTER_LINEAR
            )
        (h, w, c) = image.shape
        for i in tqdm(range(h)):
            for j in range(w):
                pixel = image[i, j]
                color.fastColorFind(pixel)
    color.plot()


main()
