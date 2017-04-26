import matplotlib.pyplot as plt
import matplotlib.image as im
import matplotlib.patches as pat

def display(img_path, boxes):
    """Display image overlayed with labled boxes

    Positional arguments:
    img_path -- local file path to background image
    boxes -- box coordinates and labes, like iter(((x,y), (w,h), l))
    """
    fig, ax = plt.subplots()
    ax.axis('off')

    ax.imshow(im.imread(img_path), cmap='gray', vmin=0, vmax=255)

    for xy, wh, label in boxes:
        rect = pat.Rectangle(xy, *wh, alpha=0.8, color='yellow')
        ax.add_patch(rect)

        rx, ry = rect.get_xy()
        cx = rx + rect.get_width()/2.0
        cy = ry + rect.get_height()/2.0
        ax.text(cx, cy, label, color='red', ha='center', va='center')

    plt.show()

def example():
    img_path = 'detection-images/detection-2.jpg'
    boxes = [
            ((113.5,71.5), (20, 20), 'M'),
            ((139.5,73.5), (20, 20), 'A'),
            ]
    display(img_path, boxes)

if __name__ == '__main__':
    example()
