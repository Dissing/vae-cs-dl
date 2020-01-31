import random

from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from math import pi, cos, sin


BLACK = 0
WHITE = 255

def create_image(name, polygons):

    img = Image.new("RGB", (64, 64))
    img_draw  = ImageDraw.Draw(img)

    masks = [ Image.new("L", (64, 64)) for _ in range(0,4) ]
    mask_draws = [ ImageDraw.Draw(img) for img in masks ]

    mask_draws[0].rectangle((0,0,64,64), WHITE)

    for i, (polygon, colour) in enumerate(polygons):
        img_draw.polygon(polygon, colour, colour)
        mask_draws[0].polygon(polygon, BLACK, BLACK)
        mask_draws[i+1].polygon(polygon, WHITE, WHITE)

    img.save(open("generated/" + name + ".png", "wb"), "PNG")
    for (i, mask) in enumerate(masks):
        mask.save(open("generated/" + name + "_" + str(i) + ".png", "wb"), "PNG")

def generate_polygon_sets(n):
    sets = []
    while len(sets) < n:
        polygons = []
        collision_polygons = []
        for i in range(3):
            size = int(round(random.random() * 3 + 4))
            shape = int(round(random.random() * 3 + 3))
            orientation = 0
            red = int(random.random() * 150) + 50
            blue = int(random.random() * 150) + 50
            green = int(random.random() * 150) + 50
            colour = (red, blue, green)

            cx = int(random.random() * 40) + 10
            cy = int(random.random() * 40) + 10
            xy = []
            for i in range(0, shape):
                a = 2*pi*i/shape + orientation
                xy.append((size * cos(a) + cx, size * sin(a) + cy))
            polygons.append((xy, colour))
            collision_polygons.append(Polygon(xy))

        collision = False

        for i, p1 in enumerate(collision_polygons):
            for j, p2 in enumerate(collision_polygons):
                if i != j and p1.intersects(p2):
                    collision = True
                    break

        if not collision:
            sets.append(polygons)

    return sets



if __name__ == "__main__":

    polygon_sets = generate_polygon_sets(2**12)

    for (idx, polygons) in enumerate(polygon_sets):
        create_image(str(idx), polygons)
