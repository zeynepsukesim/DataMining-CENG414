import sys
import kmeans
import numpy as np
from PIL import Image
import sys


def main():

    # reading pixels
    print("Reading pixels...", end="\t")
    arg_list = sys.argv[1:]

    image_file_name = arg_list[0]
    number_of_colors = int(arg_list[1])
    max_iterations = int(arg_list[2])
    epsilon = float(arg_list[3])

    image = Image.open(image_file_name)
    image_width = image.width
    image_height = image.height

    inp = []
    for i in range(image_height):
        for j in range(image_width):
            inp.append(image.getpixel((j, i)))
    # read image pixels
    # and have a list in 1 X (width * height) dimensions

    print("DONE")

    model = kmeans.KMeans(
        X=np.array(inp),
        n_clusters=number_of_colors,
        max_iterations=max_iterations,
        epsilon=epsilon,
        distance_metric="euclidian",
    )
    print("Fitting...")
    model.fit()
    print("Fitting... DONE")

    print("Predicting...")
    color1 = (134, 66, 176)
    color2 = (34, 36, 255)
    color3 = (94, 166, 126)
    print(f"Prediction for {color1} is cluster {model.predict(color1)}")
    print(f"Prediction for {color2} is cluster {model.predict(color2)}")
    print(f"Prediction for {color3} is cluster {model.predict(color3)}")

    # replace image pixels with color palette
    # (cluster centers) found in the model
    for i in range(image_height):
        for j in range(image_width):
            pixel = tuple(model.cluster_centers[model.predict(image.getpixel((j, i)))])
            pixel = (round(elem) for elem in pixel)
            pixel = tuple(pixel)
            image.putpixel((j, i), pixel)

    # save the final image
    image.show()


if __name__ == "__main__":
    main()
