import sys
import rembg
import imageio

rgb_image = imageio.imread(sys.argv[1])

result = rembg.remove(rgb_image)

imageio.imsave(sys.argv[2], result)
