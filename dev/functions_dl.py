import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import resize
import numpy as np
import pandas as pd
import cv2 as cv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import time
import six
from IPython.display import Image
from keras.preprocessing import image


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
                cmap_result='gray', filename=None):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    ax[0].imshow(original, cmap=cmap_result)
    ax[0].set_title(title_orignal)
    ax[0].axis('off')

    ax[1].imshow(transformed, cmap=cmap_result)
    ax[1].set_title(title_transformed)
    ax[1].axis('off')

    plt.subplots_adjust()
    plt.savefig('./img/preprocess_image_' + filename + '.png', transparent=True)
    plt.show()


def transform_image(original, classes=3, cmap_result='gray', filename=None):
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
    plt.savefig('./img/preprocess_image_' + filename + '_' + str(classes) + 'classes.png', transparent=True)
    plt.show()


def features(image, extractor):
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors


def get_sifts_features(image):
    descriptor_list = []
    extractor = cv.xfeatures2d.SIFT_create()
    for uri in image:
        # Lecture de l'image en N&B
        img = cv.imread(uri, 0)
        # Normalisation de l'image
        img = img.astype("float32")/255
        # Réduction du bruit
        denoised_img = denoise_tv_chambolle(img, weight=0.1, multichannel=True)
        # Conversion en 8 bit
        img8bit = cv.normalize(denoised_img,
                               None,
                               0,
                               255,
                               cv.NORM_MINMAX).astype('uint8')
        # Calcul des descripteurs
        kp, desc = features(img8bit, extractor)

        descriptor_list.extend(desc)

    return descriptor_list


def get_vector_sift(image, km, k):
    matrix = pd.DataFrame(columns=range(k))
    extractor = cv.xfeatures2d.SIFT_create()
    for uri in image:
        # Lecture de l'image en N&B
        img = cv.imread(uri, 0)
        # Normalisation de l'image
        img = img.astype("float32")/255
        # Réduction du bruit
        denoised_img = denoise_tv_chambolle(img, weight=0.1, multichannel=True)
        # Conversion en 8 bit
        img8bit = cv.normalize(denoised_img,
                               None,
                               0,
                               255,
                               cv.NORM_MINMAX).astype('uint8')
        # Calcul des descripteurs
        kp, desc = features(img8bit, extractor)

        label = km.predict(desc)
        unique, counts = np.unique(label, return_counts=True)
        matrix = matrix.append(dict(zip(unique, counts)), ignore_index=True)
    return matrix.fillna(0)


def export_png_table(data, col_width=2.2, row_height=0.625, font_size=10,
                     header_color='#7451eb', row_colors=['#f1f1f2', 'w'],
                     edge_color='w', bbox=[0, 0, 1, 1], header_columns=1,
                     ax=None, filename='table.png', **kwargs):
    ax = None
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])
                ) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox,
                         colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])

    fig.savefig(filename, transparent=True)
    plt.close()
    display(Image(filename))
    return ax


def exec_time(start, end):
    diff_time = end - start
    m, s = divmod(diff_time, 60)
    h, m = divmod(m, 60)
    s, m, h = int(round(s, 0)), int(round(m, 0)), int(round(h, 0))
    duration = "{0:02d}:{1:02d}:{2:02d}".format(h, m, s)
    return duration


def evaluate_classifier(X_train, X_test, y_train, y_test, classifiers, cv=5,
                        scoring='accuracy', target_name='models'):
    """[summary]
    Args:
        X_train (object): Données d'entrainements
        X_test (object): Données de tests
        y_train (object): Données d'entrainements
        y_test (object): Données de tests
        classifiers (dict): Contient les modèles et les hyperparamètres
        cv (int, optional): [description]. Defaults to 5.
        scoring (str, optional): [description]. Defaults to 'accuracy'.
        target_name (str, optional): [description]. Defaults to 'models'.
    """

    results = pd.DataFrame()
    for class_name, class_, class_params in classifiers:
        print(f"{class_name} en cours d'exécution...")
        model = GridSearchCV(class_, param_grid=class_params, cv=cv,
                             scoring=scoring, n_jobs=-1)
        model.fit(X_train, y_train)

        # Je stocke les résultats du GridSearchCV dans un dataframe
        model_results_df = pd.DataFrame(model.cv_results_)

        # Je sélectionne la meilleure observation
        cond = model_results_df["rank_test_score"] == 1
        model_results_df = model_results_df[cond]

        # Prediction
        y_pred = model.predict(X_test)

        # J'ajoute le nom du modéle et les résultats sur les données de test
        model_results_df[target_name] = class_name
        score = accuracy_score(y_test, y_pred)
        model_results_df['Test : ' + scoring] = round(score, 3)

        # Les hyperparamètres des classifieurs étant changeant,
        # je crée un nouveau dataframe à partir de la colonne params
        # des résultats. Je jointe les 2 dataframes à partir des index.
        col = [target_name, 'Test : ' + scoring,
               'mean_test_score', 'mean_fit_time']
        model_results_df = pd.merge(model_results_df[col],
                                    pd.DataFrame(model.cv_results_['params']),
                                    left_index=True, right_index=True)

        col = ['mean_test_score', 'mean_fit_time']
        model_results_df[col] = round(model_results_df[col], 2)
        # Je stocke les résultats dans un nouveau dataframe.
        results = results.append(model_results_df)

    filename = 'img/' + target_name + '.png'
    export_png_table(results, filename=filename)

def get_img_aug(img, datagen):
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    
    i = 0 
    y = 0
    z = 0
    for batch in datagen.flow(x):
        ax[y, z].imshow(image.array_to_img(batch[0])) 
        ax[y, z].set_title('Augmentation ' + str(i))
        # Changement de subplot
        z += 1
        if z % 2 == 0:
            z = 0
            y = 1

        # Arrêt au bout de 4 images
        i+=1
        if i % 4 == 0 :
            plt.savefig('./img/preprocess_image_data_augmentation.png', transparent=True)
            break