import itertools

from PIL import Image, ImageDraw
from math import pi, cos, sin



def create_image(name, size, shape, orientation, colour):
    im = Image.new("RGB", (64, 64))
    draw  = ImageDraw.Draw(im)

    xy = []

    for i in range(0, shape):
        a = 2*pi*i/shape + orientation
        xy.append((size * cos(a) + 32, size * sin(a) + 32))

    draw.polygon(xy, colour, colour)

    with open("generated/" + name + ".png", "wb") as f:
        im.save(f, "PNG")



if __name__ == "__main__":

    sizes = [2 * x + 15 for x in range(1,6)]
    shapes = [x for x in range(3,7)]
    orientations = [x * pi / 20.0  for x in range(0,10)]
    component = [50 * x for x in range(1, 5)]
    colours = list(itertools.product(component, component, component))

    it = itertools.product(sizes, shapes, orientations, colours)

    counter = 0

    for (idx,(size, shape, orientation, colour)) in enumerate(it):
        create_image(str(idx), size, shape ,orientation, colour)
        counter = counter + 1

    print(counter)
