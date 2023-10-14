import numpy as np
from glob import glob
from pathlib import Path
from sp_addons import AudioManipulation

import torch
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

__import__('warnings').filterwarnings("ignore")

# Каталог с датасетами
DATASET_PATH = Path(r'D:\python-datasets\dusha')
# Каталог с конкретным датасетом
EXAMPLES = DATASET_PATH.joinpath(r'ChatExport\voice_messages')
filenames = [EXAMPLES.joinpath(file) for file in glob(f'{EXAMPLES}/**', recursive=True)
             if Path(file).is_file()]
# print(*filenames, sep='\n')

# Указать устройство (GPU) для вычислений, если оно доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

extractor_name = "facebook/hubert-large-ls960-ft"
model_name = "xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(extractor_name)
model = HubertForSequenceClassification.from_pretrained(model_name)
model.to(device)

print('Вангую...')

# создаем экземпляр класса
audio_obj = AudioManipulation(feature_extractor=feature_extractor, model=model)
# тут задаем длительность фрагмента, чтобы влезло в память GPU, по умолчанию 10 сек
audio_obj.duration = 10_000

# сюда пишем полный путь к файлу
file_path = filenames[0]

try:
    # см. описание метода .predict() что в него передаем и что он возвращает
    predicts = audio_obj.predict(file_path, show_file_info=True, debug=True,
                                 simple_indexes=False)
except:
    predicts = (np.NAN, ['ошибка, не получилось обработать файл'])

print('Эмоция:', predicts[0])
print('Список эмоций по фрагментам:', predicts[1])
