import cv2
import numpy as np
from pytesseract import image_to_osd
from pytesseract import TesseractError
from deskew import determine_skew
from imgaug.augmenters import Rotate

# Удаляет черную рамку
# Хорошо работает с неповернутым изображением
def remove_black_edges(img):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    return img[y:y+h,x:x+w]


### Deskew. Хорошо работает при повороте от -90 до 90
# Если картинка повернута больше чем на 90, то находит угол для поворота вверх ногами
# И тессеракт бибилиотека определяет ориентацию изображения
# remove_black_edges может удалить всю картинку(это замечено на изображених, которые сохраняет PIL)
def deskew_full(img, min_deviation=0.1):
    angle = determine_skew(img, min_deviation=min_deviation, angle_pm_90=True)
    img = Rotate(-angle, fit_output=True, cval=0)(image=img)

    newimg = remove_black_edges(img)
    return newimg
    # try :
    #     if 'Orientation in degrees: 180' in image_to_osd(newimg):
    #         newimg = Rotate(180)(image=newimg)
    #     return newimg
    # except TesseractError:
    #     print('Препроцессинг не идеальный, поэтому это изображение с черными рамками')
    #     return img

### Удаляет тени
# На место тени приходят засвеченные области
def remove_shadow(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
        
    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)

    return result_norm


# Удаляет шум с изображения
def remove_noise(
    img: np.ndarray,
    interpolation = cv2.INTER_LINEAR,
    zoom_factor: int = 4,
    kernel_size: int = 3,
    iterations: int = 2,
    kernal_size_medianblur: int = 3
) -> np.ndarray:
    # Увеличиваем изображение
    (height, width) = img.shape[:2]
    height, width = int(height * zoom_factor), int(width * zoom_factor)
    img = cv2.resize(img, (width, height), interpolation=interpolation)
    #переводим изображение в оттенки серого
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Небольшое размытие и перевод в бинарное изображение
    img = cv2.medianBlur(img, kernal_size_medianblur)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    img = cv2.bitwise_not(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)

    height, width = int(height / zoom_factor), int(width / zoom_factor)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    return ~img


### Полный пайплайн
# deskew бывает тупит и жалуется что у картинки нулевой dpi
def preprocess(
    img,
    noise_interpolation = cv2.INTER_LINEAR,
    noise_zoom_factor = 4,
    noise_kernel_size: int = 3,
    noise_iterations: int = 2,
    noies_kernal_size_medianblur: int = 3,
    deskew_error: float = 0.1
):
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = deskew_full(img, min_deviation=deskew_error) # 2.7 sec
    img = remove_shadow(img) # 0.1 sec
    img = remove_noise( # 0.1 sec
        img,
        interpolation = noise_interpolation, 
        zoom_factor = noise_zoom_factor,
        kernel_size = noise_kernel_size,
        iterations = noise_iterations,
        kernal_size_medianblur = noies_kernal_size_medianblur
    ) 

    return img
