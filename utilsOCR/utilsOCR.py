import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor

import numpy as np
import math
import pandas as pd
from itertools import combinations, permutations

import os
import subprocess
from tqdm import tqdm

from pypdf import PdfReader as pdfreader
import fitz
import pickle

import PIL
from PIL import Image
import cv2
import imageio
import imgaug as ia
from imgaug import augmenters as iaa

from torchmetrics.text import CharErrorRate, WordErrorRate

from utilsOCR.preprocess import preprocess

# f = sorted(os.listdir('/home/boris/model_choose/data/data/img'), key=lambda x: tuple(map(int, x.rstrip('.jpg').split('-'))))
# dic = dict()
# for i, name in enumerate(f):
#     dic[i+1]



class DocDataset(Dataset):
    def __init__(self, dir, transform=None, target_transform=None,
                    store_images=False):
        """
            store_images(bool): if True, then save images in self.data as PIL.Image, else
            save as path to the each file

        """
        self.store_images = store_images
        self.data = [] 
        self.targets = []        
        if store_images:    
            for file in os.listdir(os.path.join(dir, 'img')): 
                with Image.open(os.path.join(dir, 'img', file)) as img:
                    img.load()
                    self.data.append(np.array(img))

                file = file.rstrip('.jpg').split('-')
                if len(file) < 2:
                    self.targets.append(pdfreader(os.path.join(
                        dir, 'targets', file[0] + '.pdf')).pages[0])
                else:
                    self.targets.append(pdfreader(os.path.join(
                        dir, 'targets', file[0] + '.pdf')).pages[int(file[1])-1])

        else:
            for file in os.listdir(os.path.join(dir, 'img'))[:]:
                self.data.append(os.path.join(dir, 'img', file))

                file = file.rstrip('.jpg').split('-')
                if len(file) < 2:
                    self.targets.append(pdfreader(os.path.join(
                        dir, 'targets', file[0] + '.pdf')).pages[0])
                else:
                    self.targets.append(pdfreader(os.path.join(
                        dir, 'targets', file[0] + '.pdf')).pages[int(file[1])-1])

        self.dir = dir
        self.img_dir = os.path.join(dir, 'img')
        self.target_dir = os.path.join(dir, 'targets')
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # if isinstance(idx, slice):
        #     return self.__lst[idx.start : idx.stop : idx.step]

        if not self.store_images:
            with Image.open(self.data[idx]) as img:
                img.load()
                img = np.array(img)
        else:
            img = self.data[idx]
        target = self.targets[idx].extract_text()

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
            
        img = iaa.Resize((2200, 2200))(image=img) # TODO: подобрать размер
        return img, target.ljust(10000, ' ')#torch.tensor(img, dtype=torch.uint8), 1#target

class myimg:

    augments = {
    'SaltAndPepper': [iaa.SaltAndPepper(0.05), iaa.SaltAndPepper(0.3)],
    'GaussianBlur': [iaa.GaussianBlur(sigma=1), iaa.GaussianBlur(sigma=2)],
    'MultiplyBrightness': [iaa.MultiplyBrightness(0.5), iaa.MultiplyBrightness(2.5)],
    'AddToBrightness': [iaa.AddToBrightness(70), iaa.AddToBrightness(130)],
    'GammaContrast': [iaa.GammaContrast(30), iaa.GammaContrast(100)],
    'Sharpen': [iaa.Sharpen(**{'alpha':1, 'lightness':0.6}), iaa.Sharpen(**{'alpha':1, 'lightness':0.8})],
    'Rotate': [iaa.Rotate(-7, fit_output=True), iaa.Rotate(7, fit_output=True)],
    'ScaleX': [iaa.ScaleX(0.6), iaa.ScaleX(1.5)],
    'ScaleY': [iaa.ScaleY(0.6), iaa.ScaleY(1.5)],
    'PiecewiseAffine': [iaa.PiecewiseAffine(0.005), iaa.PiecewiseAffine(0.03)],
    'Pixelate': [iaa.imgcorruptlike.Pixelate(1), iaa.imgcorruptlike.Pixelate(4)],
    'Resize': [iaa.Resize((900, 1300)), iaa.Resize((1300, 1600))],

    'BlendAlphaHorizontalLinearGradient': [iaa.BlendAlphaHorizontalLinearGradient(
                                            **{'foreground':iaa.TotalDropout(1.0), 'max_value': 0.8, 'min_value': 0.1}),
                                            iaa.BlendAlphaHorizontalLinearGradient(
                                            **{'foreground':iaa.TotalDropout(1.0), 'max_value': 0.8, 'min_value': 0.8})],

    'BlendAlphaVerticalLinearGradient': [iaa.BlendAlphaVerticalLinearGradient(
                                            **{'foreground':iaa.TotalDropout(1.0), 'max_value': 0.8, 'min_value': 0.1}),
                                            iaa.BlendAlphaHorizontalLinearGradient(
                                            **{'foreground':iaa.TotalDropout(1.0), 'max_value': 0.8, 'min_value': 0.8})]
    }

    # Применяет aug к img несколько раз и показывает результаты(TODO)
    def check_aug(self, aug, img):
        arr = [aug(image=img) for i in np.linspace(70, 150, 6)]
        ia.imshow(np.concatenate(arr))

    # Приводит np.array к заданному размеру size
    def to_size(self, arr: np.array, size):

        shape = arr.shape
        arr = np.concatenate([arr, np.zeros((shape[0], size[1] - shape[1]), dtype=arr.dtype)], axis=1)
        arr = np.concatenate([arr, np.zeros((size[0] - shape[0], size[1]), dtype=arr.dtype)], axis=0)
        return arr

    def save_augments(self, augments: dict, path: str, img: np.array, only_one_aug=True):
        """
            Применяет каждую аугментацию из augment и сохраняет в свой пдф файл
            все ее вариации приставленные в augment. Если only_one_aug=False, то 
            берет комбинации по 2 в разном порядке

                augments: словарь, где ключ это название аугментации, а значение - список 
                            из этой аугментации с разными параметрами

                path: путь куда сохранять 

                img: изображение на котором производятся аугментации

                only_one_aug: если True, то применяется ровно одна аугментация
                            если False, то применяются две аугментации во всемозможных
                            комбинациях и порядках

        """
        if only_one_aug:
            for aug_name in tqdm(augments.keys()):

                img1 = augments[aug_name][0](image=img)
                img2 = augments[aug_name][1](image=img)
                    
                max_shape = max(img1.shape, img2.shape)
                # print((img1.shape, img2.shape, max_shape))
                imgs = iaa.PadToFixedSize(*max_shape[:-1])(images=[img1, img2])
                # print((imgs[0].shape, imgs[1].shape, max_shape))
                Image.fromarray(np.concatenate(imgs, axis=1)).save(os.path.join(path, aug_name + '.pdf'))
        else:
            for aug1_name, aug2_name in tqdm(combinations(augments.keys(), 2)):

                aug10, aug11 = augments[aug1_name][0], augments[aug1_name][1]
                aug20, aug21 = augments[aug2_name][0], augments[aug2_name][1]
                aug_combs = set(permutations([aug10, aug11, aug20, aug21], 2)) - set(
                                    [(aug10, aug11), (aug11, aug10), (aug20, aug21), (aug21, aug20)])
                
                imgs = []
                for aug1, aug2 in aug_combs:
                    seq = iaa.Sequential([aug1, aug2])
                    imgs.append(seq(image=img))
                    
                max_shape = max(map(np.shape, imgs))
                imgs = iaa.PadToFixedSize(*max_shape[:-1], position='right-bottom')(images=imgs)
                Image.fromarray(np.concatenate(imgs, axis=1)).save(os.path.join(path,
                                                aug1_name + ' + ' + aug2_name + '.pdf'))
        
# Боксы в формате [tl, tr, br, bl] в координатах x,y
def draw_ocr(image,
             boxes,
             txts=None,
             scores=None,
             drop_score=0.5,
             font_path="./doc/simfang.ttf"):
    """
    Visualize the results of OCR detection and recognition
    args:
        image(Image|array): RGB image
        boxes(list): boxes with shape(N, 4, 2)
        txts(list): the texts
        scores(list): txxs corresponding scores
        drop_score(float): only scores greater than drop_threshold will be visualized
        font_path: the path of font which is used to draw text
    return(array):
        the visualized img
    """
    if scores is None:
        scores = [1] * len(boxes)
    box_num = len(boxes)
    for i in range(box_num):
        if scores is not None and (scores[i] < drop_score or
                                   math.isnan(scores[i])):
            continue
        box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    return image

# Боксы в формате [tl, tr, br, bl] в координатах x,y переводит в бокс заданный отступами от границ
# И если бокс повернутый, то возвращает бокс параллельный границам
def crd2len(x):
    return [ min(x[0][0], x[3][0]), min(x[0][1], x[1][1]), max(x[1][0], x[2][0]), max(x[2][1], x[3][1]) ]

# Обратно к crd2len
def len2crd(x):
    return [ [x[0], x[1]], [x[2], x[1]], [x[2], x[3]], [x[0], x[3]] ]

# xywh -> отступы от границ
def yolo2len(x, width, height):
    return [ 
        (x[0] - x[2]/2)*width, 
        (x[1] - x[3]/2)*height, 
        (x[0] + x[2]/2)*width, 
        (x[1] + x[3]/2)*height
        ]

# Вырезает боксы из картинки и сохраняет их в папку
def crop(boxes, img_path, output_path):
    with Image.open(img_path) as img:
        img.load()

    for i, box in enumerate(boxes):
        img.crop(crd2len(box)).save(output_path + f'/{i}.jpg', "JPEG")

# Выводит изображение
def show(img: np.array):
    ia.imshow(img)

# Переводит pil.image в тензор формата HWC
def pil2tensor(img):
    return np.transpose(pil_to_tensor(img), (1,2,0))

# Переводит pdf в numpy.ndarray RGB
def pdf2numpy(file_path, page=0, zoom=3):
    doc = fitz.open(file_path)
    page = doc[page]

    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)


AUGMENTS=[
    iaa.BlendAlphaHorizontalLinearGradient(
        iaa.TotalDropout(1.0),
        min_value=(0.1, 0.8),
        max_value=(0.1, 0.8)),
    iaa.BlendAlphaVerticalLinearGradient(
        iaa.TotalDropout(1.0),
        min_value=(0.1, 0.8),
        max_value=(0.1, 0.8)),
    iaa.SaltAndPepper((0.01, 0.1)),
    iaa.MultiplyBrightness((0.5, 1.2)),
    iaa.AddToBrightness((20, 50)), 
    iaa.GammaContrast((15, 35)),
    iaa.Rotate((-7,7), fit_output=True),
    iaa.ScaleX((0.7, 1.4), fit_output=True),
    iaa.ScaleY((0.7, 1.4), fit_output=True),
    iaa.imgcorruptlike.Pixelate([1, 2]),
    # iaa.GaussianBlur(sigma=(1, 2)),
    # iaa.Sharpen(alpha=1, lightness=(0.6, 0.8)),
    # iaa.PiecewiseAffine((0.005, 0.01)),
    ]
# Применяет случайную аугментацию к img из augs 
def one_rand_aug(img: np.array, augs=AUGMENTS):
    return np.random.choice(augs)(image=img)

# Чтобы не приходилось писать image=
def app_aug(aug):
    return lambda img: aug(image=img)

# Принимает путь к pdf и номер страницы, и сохраняет боксы слов в save_dir
# в формате нужном для тренеровки YOLO
# TODO: куда-то сохранять слова которые лежат в боксах
def pdf_to_yoloformat(file_path, num_page=0, save_dir=None):
    doc = fitz.open(file_path)
    page = doc[num_page]
    tp = page.get_textpage()

    bboxes = tp.extractWORDS()
    bboxes = np.array(bboxes)[:, :5]

    words = bboxes[:, 4] #### TODO
    boxes = bboxes[:, :4].astype('float32')
    boxes = np.array([[(b[0], b[3]), (b[0], b[1]), (b[2], b[3]), (b[2], b[1])] for b in boxes])

    width, height = page.rect[2], page.rect[3]

    text = ''
    for box in boxes:
        x, y = np.round(((box[0] + box[3]) / 2) / [width, height], 6)
        w, h = np.round((box[3] - box[0]) / [width, -height], 6)
        text += f"0 {x} {y} {w} {h}\n"

    if save_dir == None:
        return text, words
    else:
        file_name = os.path.basename(file_path).split('.')[0] + f'_{num_page}.txt'
        with open(os.path.join(save_dir, file_name), 'w') as file:
            file.write(text)

def pdf_to_craftformat(
    file_path, 
    page=0, 
    save_dir=None,
    transform = lambda x: iaa.Identity()(image=x),
    train_test = 'train',
    make_extra_folders = False
):
    if make_extra_folders:
        os.mkdir(path = os.path.join(save_dir, 'ch4_training_images'))
        os.mkdir(path = os.path.join(save_dir, 'ch4_training_localization_transcription_gt'))
        os.mkdir(path = os.path.join(save_dir, 'ch4_test_images'))
        os.mkdir(path = os.path.join(save_dir, 'ch4_test_localization_transcription_gt'))
        print('Folders has been made. If you want to add files, do \"make_extra_folders = False\"')
        return

    bboxes, words = pdf_to_yoloformat(file_path, num_page=page)

    img = pdf2numpy(file_path, page)
    img = transform(img)
    img = Image.fromarray(img.astype('uint8'))

    w, h = img.width, img.height
    bboxes = [yolo2len(list(map(float, box.split()[1:])), w, h) for box in bboxes.split('\n')[:-1]]
    bboxes = [len2crd(box) for box in bboxes]
    bboxes = np.array(bboxes, dtype=np.int32).reshape((-1, 8))
    bboxes = bboxes.astype(str)

    text = ''
    for box, word in zip(bboxes, words):
        text += ','.join(box) + f',{word}\n'

    if save_dir == None:
        return text
    elif train_test == 'train':
        file_name = os.path.basename(file_path).split('.')[0]
        with open(os.path.join(
                    save_dir,
                    'ch4_training_localization_transcription_gt',
                    'gt_' + file_name + '.txt'
                    ), 'w') as file:
            file.write(text)
        img.save(os.path.join(save_dir, 'ch4_training_images', file_name + '.jpg'), 'JPEG')
    elif train_test == 'test':
        file_name = os.path.basename(file_path).split('.')[0]
        with open(os.path.join(
                    save_dir,
                    'ch4_test_localization_transcription_gt',
                    'gt_' + file_name + '.txt'
                    ), 'w') as file:
            file.write(text)
        img.save(os.path.join(save_dir, 'ch4_test_images', file_name + '.jpg'), 'JPEG')
    else:
        raise ValueError(f'train_test can be only \"train\" or \"test\", but got \"{train_test}\"')

# Принимает путь файлу с боксами в формате pickle, а также ширину и высоту изображения
# и сохраняет в save_dir в формате нужном для тренеровки YOLO
# !!! Файлы на вход должны быть такие же как в датасете ddi-100
def ddi100_to_yoloformat(file_path, width, height, save_dir=None):
    objects = []
    with (open(file_path, "rb")) as f:
        while True:
            try:
                objects.append(pickle.load(f))
            except EOFError:
                break

    boxes = np.array([obj['box'] for obj in objects[0]])
    boxes[:,:, [0, 1]] = boxes[:,:, [1, 0]]
    text = ''
    for box in boxes:
        x, y = np.round(((box[0] + box[3]) / 2) / [width, height], 6)
        w, h = np.round((box[3] - box[0]) / [width, -height], 6)
        text += f"0 {x} {y} {w} {h}\n"

    if save_dir == None:
        return text
    else:
        file_name = os.path.basename(file_path).split('.')[0] + '.txt'
        with open(os.path.join(save_dir, file_name), 'w') as file:
            file.write(text)

# Принимает путь к pdf и номер страницы.
# Сохраняет картинки вырезанных боксов с аннотицией в save_dir
# !!! transform которые портят боксы с относительными координатами не подходят
# transform = transforms.Compose([
#     one_rand_aug,
#     preprocess
# ])
def pdf_to_recognizer_train_format(
    file_path,
    page = 0, 
    save_dir = None, 
    transform = lambda x: iaa.Identity()(image=x),
    image_format = 'jpg'
):

    bboxes, words = pdf_to_yoloformat(file_path, num_page=page)

    img = pdf2numpy(file_path, page)
    img = transform(img)
    img = Image.fromarray(img.astype('uint8'))

    w, h = img.width, img.height
    bboxes = [yolo2len(list(map(float, box.split()[1:])), w, h) for box in bboxes.split('\n')[:-1]]

    file_name = os.path.basename(file_path).split('.')[0]
    names = [f'{file_name}_{page}_{i}.{image_format}' for i in range(len(words))]
    df = pd.DataFrame({
        'filename': names,
        'words': words
        })

    if save_dir == None:
        return bboxes, img, df
    else:
        if image_format.lower() == 'jpg':
            for i, box in enumerate(bboxes):
                img.crop(box).save(os.path.join(save_dir, f'{file_name}_{page}_{i}.jpg'), "JPEG")
        elif image_format.lower() == 'png':
            for i, box in enumerate(bboxes):
                img.crop(box).save(os.path.join(save_dir, f'{file_name}_{page}_{i}.png'), "PNG")
        else:
            raise KeyError
        df.to_csv(os.path.join(save_dir, 'labels.csv'), index=False)


def pdf_to_tesseract_train_format(
    file_path,
    save_dir,
    page = 0,
    transform = lambda x: iaa.Identity()(image=x)
):
    bboxes, img, df = pdf_to_recognizer_train_format(
        file_path = file_path,
        page = page,
        transform = transform,
        image_format = 'png'
    )

    try:
        df_old = pd.read_csv(os.path.join(save_dir, 'labels.csv'))
    except FileNotFoundError:
        df_old = pd.DataFrame({'filename': [], 'words': []})
    
    if inter := (set(df['filename']) & set(df_old['filename'])):
        raise Exception(f'In {save_dir} there is already such files as {inter}')
    
    for i, box in enumerate(bboxes):
        img.crop(box).save(os.path.join(save_dir, f'{df['filename'][i]}'), "PNG")
    
    for name, word in zip(df['filename'], df['words']):
        with open(os.path.join(save_dir, f'{name[:-4]}.gt.txt'), 'w') as file:
            file.write(word + '\n') ###############

    df_merged = pd.concat([df_old, df], ignore_index=True, sort=False)
    df_merged.to_csv(os.path.join(save_dir, 'labels.csv'), index=False)





