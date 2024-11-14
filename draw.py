import cv2
import numpy as np

def main(args):
    img = cv2.imread(args.filename, cv2.IMREAD_COLOR)
    with open(args.annotation, 'r') as fp:
        first_line = fp.readline().rstrip()
        cx, cy, h, a = [int(i) for i in first_line.split()]

    sina = np.sin(np.pi*a/180.0)
    cosa = np.cos(np.pi*a/180.0)

    vx, vy = -sina*h/2.0, cosa*h/2.0
    hx, hy = cosa*0.6*h/2.0, sina*0.6*h/2.0

    cv2.line(img, (cx,cy), (int(cx+vx), int(cy+vy)), (0,255,0), 3)
    cv2.line(img, (cx,cy), (int(cx+hx), int(cy+hy)), (0,0,255), 3)

    cv2.line(img, (int(cx-vx-hx),int(cy-vy-hy)), (int(cx-vx+hx), int(cy-vy+hy)), (255,0,0), 3)
    cv2.line(img, (int(cx-vx+hx),int(cy-vy+hy)), (int(cx+vx+hx), int(cy+vy+hy)), (255,0,0), 3)
    cv2.line(img, (int(cx+vx+hx),int(cy+vy+hy)), (int(cx+vx-hx), int(cy+vy-hy)), (255,0,0), 3)
    cv2.line(img, (int(cx+vx-hx),int(cy+vy-hy)), (int(cx-vx-hx), int(cy-vy-hy)), (255,0,0), 3)
    
    # Resize the image to make the screen size smaller
    scale_percent = 50  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow('image', resized)
    cv2.waitKey()


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(prog='draw', description='Draw an oriented bounding box')
	parser.add_argument('-f', '--filename', help='Input image file.')
	parser.add_argument('-a', '--annotation', help='Input annotation file.')
	args = parser.parse_args()

	main(args)
