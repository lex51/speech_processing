import os
from celery import Celery
from loguru import logger as lg

import numpy as np
from pathlib import Path
from sp_addons import AudioManipulation

import torch
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

__import__("warnings").filterwarnings("ignore")

print(torch.__version__)  # 1.12.1+cpu
print(__import__("transformers").__version__)  # 4.29.2

passcode = os.environ.get("REDIS_PASS")
CELERY_BROKER_URL = (
    os.environ.get("CELERY_BROKER_URL", "redis://:" + passcode + "@localhost:6379"),
)
CELERY_RESULT_BACKEND = os.environ.get(
    "CELERY_RESULT_BACKEND", "redis://:" + passcode + "@localhost:6379"
)

celery = Celery("tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)


# Каталог с датасетами
DATASET_PATH = Path("/datasets/dusha/")
# Каталог с конкретным датасетом
# EXAMPLES = DATASET_PATH.joinpath(r"ChatExport\voice_messages")
extractor_name = "facebook/hubert-large-ls960-ft"
model_name = "xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned"


@celery.task(name="tasks.get_emoji")
def get_emoji(speech_obj):
    # Каталог с датасетами

    # print(*filenames, sep='\n')

    # Указать устройство (GPU) для вычислений, если оно доступно
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(extractor_name)
    model = HubertForSequenceClassification.from_pretrained(model_name)
    model.to(device)

    # print(f"Вангую на {device} ...")
    #
    # создаем экземпляр класса
    audio_obj = AudioManipulation(feature_extractor=feature_extractor, model=model)
    # тут задаем длительность фрагмента, чтобы влезло в память GPU, по умолчанию 10 сек
    audio_obj.duration = 10_000

    try:
        # см. описание метода .predict() что в него передаем и что он возвращает
        predicts = audio_obj.predict(
            speech_obj, show_file_info=True, debug=True, simple_indexes=False
        )
    except:
        predicts = (np.NAN, ["ошибка, не получилось обработать файл"])

    lg.debug(f"main_emo= {predicts[0]}")
    lg.debug(f"list_emo= {predicts[1]}")
    return {"main_emo": predicts[0], "info_by_periods": predicts[1]}
