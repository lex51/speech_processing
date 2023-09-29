import tempfile
import pandas as pd
import torch
from math import ceil
from glob import glob
from pathlib import Path
from statistics import mode as stat_mode

from pydub import AudioSegment
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

__import__('warnings').filterwarnings("ignore")

import torchaudio


class AudioManipulation:
    def __init__(self, feature_extractor=None, model=None):
        # поддерживаемые конвертеры
        self.converters = {'.ogg': AudioSegment.from_ogg,
                           '.mp3': AudioSegment.from_mp3,
                           '.wav': AudioSegment.from_wav}
        # Получение пути к временному файлу
        self.temp_file_name = f'{tempfile.NamedTemporaryFile(delete=True).name}.wav'
        # минимальная длительность аудиофайла 1 секунда
        self.audio_min_duration = 1_000
        # максимальная длительность аудиофайла 10 минут
        self.audio_max_duration = 60_000 * 10
        # будем предсказывать на фрагментах по ХХ секунд
        self.duration = 10_000
        self.feature_extractor = feature_extractor
        self.model = model
        # Указать устройство (GPU) для вычислений, если оно доступно
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # словарь эмоций
        self.num2emotion = {0: 'neutral', 1: 'angry', 2: 'positive', 3: 'sad', 4: 'other'}
        # предел недовольных фрагментов в долях от общего количества
        self.angry_threshold = .2

    def convert_to_wav(self, path_to_file, max_duration=None, show_file_info=False):
        """
        Функция принимает полный путь к файлу и если это не .wav - конвертирует в .wav,
        сохраняет во временный файл и возвращает путь к этому файлу
        :param path_to_file: полный путь к файлу
        :param max_duration: максимальная длительность аудиофайла
        :param show_file_info: Печать информацию о файле: имя и длительность
        :return: полный путь к .wav файлу и звуковой файл в формате .wav
        """
        if not Path(path_to_file).is_file():
            raise FileNotFoundError(f"Ошибка!!! Файл {path_to_file} не найден!")

        suffix = Path(str(path_to_file).lower()).suffix
        converter = self.converters.get(suffix)
        if converter is None:
            raise TypeError("Ошибка!!! Не поддерживаемый формат файла!")

        sound = converter(path_to_file)

        if show_file_info:
            print(f'Файл: {Path(path_to_file).name}',
                  f'Длительность: {round(len(sound) / 1000., 1)} сек')

        if max_duration is not None:
            sound = sound[:max_duration]
        sound.export(self.temp_file_name, format="wav")
        return self.temp_file_name, sound

    def get_indexes(self, sound):
        """
        Получение индексов фрагментов аудио файла
        :param sound: экземпляр AudioSegment
        :return: список индексов
        """
        # Это самый простой способ получения списков индексов: поделить аудио файл на части
        # заданной длительности self.duration
        sound_parts = int(ceil(len(sound) / self.duration))
        indexes = [self.duration * idx for idx in range(sound_parts)]
        # Если хвост аудио файла больше минимальной длительности -> добавим еще индекс
        if len(sound) - indexes[-1] > self.audio_min_duration:
            indexes.append(len(sound) + 1)
        return indexes

    def predict(self, path_to_file, max_duration=None, debug=False, show_file_info=False):
        """
        Предсказание эмоции для одного аудио файла
        :param path_to_file: полный путь к файлу
        :param max_duration: максимальная длительность аудиофайла
        :param debug: debug=True - режим отладки
        :param show_file_info: Печать информацию о файле: имя и длительность
        :return: основная эмоция, список всех эмоций
        """
        temp_file_name, sound = self.convert_to_wav(path_to_file,
                                                    max_duration=max_duration,
                                                    show_file_info=show_file_info)

        temp_wav = str(temp_file_name).replace('.wav', '_part.wav')
        if debug:
            print(f'Временный файл: {temp_wav}')

        file_emotions = []

        # старая версия разбиения на части
        # sound_parts = int(ceil(len(sound) / self.duration))
        # for idx in range(sound_parts):
        #     if debug:
        #         print(f'Часть: {idx+1}', (idx * self.duration, (idx+1) * self.duration))
        #
        #     part_sound = sound[idx * self.duration:(idx + 1) * self.duration]

        indexes = self.get_indexes(sound)
        sound_parts = len(indexes) - 1
        for idx, start_stop in enumerate(zip(indexes, indexes[1:])):
            idx_start, idx_stop = start_stop
            if debug:
                print(f'Часть: {idx + 1}', (idx_start, idx_stop))
            part_sound = sound[idx_start: idx_stop]

            # длительность меньше порога - не будем предсказывать
            part_len = len(part_sound)
            if part_len < self.audio_min_duration:
                if debug:
                    print(f'Обнаружен фрагмент с длительностью {part_len} менее допустимой')
                continue

            part_sound.export(temp_wav, format="wav")

            waveform, sample_rate = torchaudio.load(temp_wav, normalize=True)
            transform = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = transform(waveform)

            # Перенести данные на GPU
            waveform = waveform.to(self.device)

            # если определены экстрактор и модель - будем считать
            # if self.feature_extractor is not None and self.model is not None:
            if all(map(bool, (self.feature_extractor, self.model))):
                inputs = self.feature_extractor(
                    waveform,
                    sampling_rate=self.feature_extractor.sampling_rate,
                    return_tensors="pt",
                    padding=True,
                    max_length=16000 * 10,
                    truncation=True
                )
                # в этом месте на 2-х файлах была ошибка - нужно исследовать
                try:
                    # Перенести данные на GPU
                    inputs = {key: val.to(self.device) for key, val in inputs.items()}

                    logits = self.model(inputs['input_values'][0]).logits
                    predictions = torch.argmax(logits, dim=-1)
                    # Перенести результаты на CPU
                    pred = predictions.cpu().numpy()[0]
                except:
                    pred = -1
            else:
                pred = 0

            predicted_emotion = self.num2emotion.get(pred)
            if debug:
                print('Предсказана эмоция:', predicted_emotion)
            file_emotions.append(predicted_emotion)

        # ищем моду эмоций
        pred = stat_mode(file_emotions)
        # считаем кол-во "недовольных" частей
        angry = file_emotions.count('angry')
        # если недовольных частей более self.angry_threshold %
        if pred != 'angry' and angry / sound_parts >= self.angry_threshold:
            pred = 'angry'
        if not all(map(bool, (self.feature_extractor, self.model))):
            print('Внимание!!! feature_extractor и model не заданы!')
        return pred, file_emotions


if __name__ == "__main__":
    audio_obj = AudioManipulation()
