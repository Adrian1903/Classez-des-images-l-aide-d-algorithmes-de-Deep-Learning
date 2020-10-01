import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu


def resize_h220(img, ratio):
    ratio = ratio
    width = 220
    height = round(width * ratio, 0)

    return resize(img, (height, width), anti_aliasing=True)


def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()


def rgb_color(img):
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    ax[0, 0].imshow(img)
    ax[0, 0].set_title('Original')
    ax[0, 0].axis('off')

    ax[0, 1].hist(red.ravel(), bins=256, color='red')
    ax[0, 1].set_title('Red histogram')

    ax[1, 0].hist(green.ravel(), bins=256, color='green')
    ax[1, 0].set_title('Blue histogram')

    ax[1, 1].hist(blue.ravel(), bins=256, color='blue')
    ax[1, 1].set_title('Green histogram')

    plt.subplots_adjust()
    plt.show()


def compare_img(original, transformed, title_orignal='Original',
                title_transformed='Transformed',
                cmap_result='gray'):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    ax[0].imshow(original, cmap=cmap_result)
    ax[0].set_title(title_orignal)
    ax[0].axis('off')

    ax[1].imshow(transformed, cmap=cmap_result)
    ax[1].set_title(title_transformed)
    ax[1].axis('off')

    plt.subplots_adjust()
    plt.show()


def transform_image(original, classes=3, cmap_result='gray'):
    thresholds = threshold_multiotsu(original, classes=classes)
    regions = np.digitize(original, bins=thresholds)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

    # Plotting the original image.
    ax[0].imshow(original, cmap='gray')
    ax[0].set_title('Grayscale')
    ax[0].axis('off')

    # Plotting the histogram and the two thresholds obtained from
    # multi-Otsu.
    ax[1].hist(original.ravel(), bins=255)
    ax[1].set_title('Histogram')
    for thresh in thresholds:
        ax[1].axvline(thresh, color='r')

    # Plotting the Multi Otsu result.
    ax[2].imshow(regions, cmap=cmap_result)
    ax[2].set_title(str(classes) + ' gray levels')
    ax[2].axis('off')

    plt.subplots_adjust()

    plt.show()
