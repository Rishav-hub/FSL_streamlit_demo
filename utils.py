import matplotlib.patches as patches
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
import io

import streamlit as st


with open('notes.json') as f:
    category_mapping = {c['id'] + 1: c['name'] for c in json.load(f)['categories']}

# def plot_img_bbox(img, target):
#     # plot the image and bboxes
#     # Bounding boxes are defined as follows: x-min y-min width height
#     fig, a = plt.subplots(1,1)
#     fig.set_size_inches(50,50)
#     img = img.permute(1, 2, 0)
#     a.imshow(img)
#     for i, box in enumerate(target['boxes'].detach().cpu().numpy()):
#         x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
#         rect = patches.Rectangle((x, y),
#                                  width, height,
#                                  linewidth = 2,
#                                  edgecolor = 'r',
#                                  facecolor = 'none')
#         # Draw the bounding box on top of the image
#         a.add_patch(rect)

#         # Add text label with class name
#         class_name = category_mapping[target['labels'][i].item()]
#         a.text(x, y, class_name, fontsize=22, color='white', verticalalignment='top', bbox={'color': 'red', 'alpha': 0.5, 'pad': 0})

#     plt.show()


def plot_bounding_boxes(image, df, enable_title = False):
    # Load the image
    # image = Image.open(image_path)

    # Get the size of the image
    image_width, image_height = image.size

    # Define a list of colors
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'magenta', 'brown']

    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(50, 50))

    # Set aspect ratio to 'auto' for full scale display
    ax.set_aspect('auto')

    # Display the image
    ax.imshow(image)

    # Iterate over the rows of the dataframe and plot the bounding boxes
    for index, row in df.iterrows():
        class_name = row['class_name']
        x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']

        # Randomly select a color from the list
        box_color = random.choice(colors)

        # Create a Rectangle patch with color
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=3, edgecolor=box_color, facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        if enable_title:

          # Add text with matching color
          ax.text(x0, y0, class_name, color=box_color, fontsize=22, weight='bold')

    # Remove axis
    ax.axis('off')

    # Show plot
    st.pyplot(fig)

def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)