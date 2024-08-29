#  OCR Documents

Ссылка на демо: https://colab.research.google.com/drive/1EVBR5fGD2X94jIlHMkUthP5sZ6Fu5dT6?usp=sharing
## Pipeline

1) Картинку идет в препроцессинг\
     а) [Делается поворот](https://github.com/sbrunner/deskew) \
     б) [Удалются тени](https://stackoverflow.com/questions/44752240/how-to-remove-shadow-from-scanned-images-using-opencv) \
     в) [Удаляется шум](https://stackoverflow.com/questions/62042172/how-to-remove-noise-in-image-opencv-python)
   
2) Предобработанная картинка отправляется в детектор, который ищет bounding boxes слов и символов
3) После вырезанные боксы отправляются распознавателю, который в каждом боксе должен распознать слово\символ

В итоге имеем: коориднаты боксов, слова в боксах и конфиденц модели для каждого слова

## Модели

### __Детектор__: [YOLOv8](https://docs.ultralytics.com/models/yolov8/)
Файнтинился на документах прошедших препроцессинг\
P.S. Перед этим дополнительно файнтюнился на разных книгах и синтетическом датасете русских слов, чтобы детектить слова.

### __Распознаватель__: [TrOCR](https://huggingface.co/Stealer0/trocr-base-ru_docs): [[paper](https://arxiv.org/abs/2109.10282)]
Файнтинился на документах прошедших препроцессинг

