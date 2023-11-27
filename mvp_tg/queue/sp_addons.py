import tempfile
import torch
from math import ceil
from pathlib import Path
from statistics import mode as stat_mode

from pydub import AudioSegment
from pydub.silence import split_on_silence

__import__("warnings").filterwarnings("ignore")

import torchaudio


class AudioManipulation:
    def __init__(self, feature_extractor=None, model=None):
        # поддерживаемые конвертеры
        self.converters = {
            ".ogg": AudioSegment.from_ogg,
            ".mp3": AudioSegment.from_mp3,
            ".wav": AudioSegment.from_wav,
            ".flv": AudioSegment.from_flv,
        }
        # Получение пути к временному файлу
        self.temp_file_name = f"{tempfile.NamedTemporaryFile(delete=True).name}.wav"
        # минимальная длительность аудиофайла 600 миллисекунд
        self.audio_min_duration = 600
        # максимальная длительность аудиофайла 10 минут
        self.audio_max_duration = 60_000 * 10
        # будем предсказывать на фрагментах по ХХ секунд
        self.duration = 10_000
        self.feature_extractor = feature_extractor
        self.model = model
        # Указать устройство (GPU) для вычислений, если оно доступно
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # словарь эмоций
        self.num2emotion = {
            0: "neutral",
            1: "angry",
            2: "positive",
            3: "sad",
            4: "other",
        }
        # предел недовольных фрагментов в долях от общего количества
        self.angry_threshold = 0.2
        self.min_silence_len = 100
        self.silence_thresh = -24

    def convert_to_wav(self, path_to_file, max_duration=None, show_file_info=False):
        """
        Функция принимает полный путь к файлу и если это не .wav - конвертирует в .wav,
        сохраняет во временный файл и возвращает путь к этому файлу
        :param path_to_file: полный путь к файлу
        :param max_duration: максимальная длительность аудиофайла
        :param show_file_info: Печать информацию о файле: имя и длительность
        :return: полный путь к .wav файлу и файл в формате .wav (экземпляр AudioSegment)
        """
        if not Path(path_to_file).is_file():
            raise FileNotFoundError(f"Ошибка!!! Файл {path_to_file} не найден!")

        suffix = Path(str(path_to_file).lower()).suffix
        converter = self.converters.get(suffix, AudioSegment.from_file)
        if converter is None:
            raise TypeError("Ошибка!!! Не поддерживаемый формат файла!")

        sound = converter(path_to_file)

        if show_file_info:
            print(
                f"Файл: {Path(path_to_file).name}",
                f"Длительность: {round(len(sound) / 1000., 1)} сек",
            )

        if max_duration is not None:
            sound = sound[:max_duration]

        # Нормализация аудио
        sound = sound.normalize()

        sound.export(self.temp_file_name, format="wav")
        return self.temp_file_name, sound

    def split_audio_to_words(self, sound, min_silence_len=None, silence_thresh=None):
        """
        Разделение звукового файла на фрагменты по паузам
        :param sound: экземпляр AudioSegment
        :param min_silence_len:
        :param silence_thresh:
        :return:
        """
        # Разделение звукового файла на фрагменты по паузам
        if min_silence_len is None:
            min_silence_len = self.min_silence_len
        if silence_thresh is None:
            silence_thresh = self.silence_thresh

        chunks = split_on_silence(
            sound,
            keep_silence=True,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
        )
        words_indexes = []
        current_time = 0
        # Перебор фрагментов и получение длительности каждого слова в миллисекундах
        for chunk in chunks:
            duration = len(chunk)
            words_indexes.append((current_time, duration))
            current_time += duration

        return words_indexes

    def get_indexes(self, sound, simple_indexes=True):
        """
        Получение индексов фрагментов аудио файла
        :param sound: экземпляр AudioSegment
        :param simple_indexes: Формировать индексы простым способом
        :return: список индексов
        """
        if simple_indexes:
            # Это самый простой способ получения списков индексов: поделить аудио файл
            # на части заданной длительности self.duration
            sound_parts = int(ceil(len(sound) / self.duration))
            indexes = [self.duration * idx for idx in range(sound_parts)]
            # Если хвост аудио файла больше минимальной длительности -> добавим еще индекс
            if len(sound) - indexes[-1] > self.audio_min_duration:
                indexes.append(len(sound) + 1)
        else:
            words_indexes = self.split_audio_to_words(sound)
            indexes = [0]
            current_time = 0
            words_duration = 0
            # Перебор фрагментов и получение длительности каждого слова в миллисекундах
            for start, duration in words_indexes:
                if words_duration + duration < self.duration:
                    words_duration += duration
                else:
                    current_time += words_duration
                    # Если длительность фрагмента > заданной длительности self.duration и
                    # остаток до self.duration > 30% -> отрежем от фрагмента кусок
                    if (
                        duration > self.duration
                        and words_duration / self.duration < 0.7
                    ):
                        words_duration = self.duration - words_duration
                        current_time += words_duration
                        duration -= words_duration
                    indexes.append(current_time)
                    words_duration = duration
                    # пока фрагмент больше self.duration -> будем резать его на куски
                    # заданной длительности self.duration
                    while words_duration > self.duration:
                        current_time += self.duration
                        indexes.append(current_time)
                        words_duration -= self.duration
            # если "хвостик" меньше минимальной длительности фрагмента ->
            # прибавим его к последнему фрагменту
            if words_duration < self.audio_min_duration and len(indexes) > 1:
                indexes.pop()
            indexes.append(current_time + words_duration + 1)
        return indexes

    def predict(
        self,
        path_to_file,
        simple_indexes=True,
        max_duration=None,
        debug=False,
        show_file_info=False,
    ):
        """
        Предсказание эмоции для одного аудио файла
        :param path_to_file: полный путь к файлу
        :param simple_indexes: Формировать индексы простым способом (True) или по паузам
        :param max_duration: максимальная длительность аудиофайла
        :param debug: debug=True - режим отладки
        :param show_file_info: Печать информацию о файле: имя и длительность
        :return: основная эмоция, список всех эмоций
        """
        temp_file_name, sound = self.convert_to_wav(
            path_to_file, max_duration=max_duration, show_file_info=show_file_info
        )

        temp_wav = str(temp_file_name).replace(".wav", "_part.wav")
        if debug:
            print(f"Временный файл: {temp_wav}")

        file_emotions = []

        # старая версия разбиения на части
        # sound_parts = int(ceil(len(sound) / self.duration))
        # for idx in range(sound_parts):
        #     if debug:
        #         print(f'Часть: {idx+1}', (idx * self.duration, (idx+1) * self.duration))
        #
        #     part_sound = sound[idx * self.duration:(idx + 1) * self.duration]

        indexes = self.get_indexes(sound, simple_indexes=simple_indexes)
        sound_parts = len(indexes) - 1
        for idx, start_stop in enumerate(zip(indexes, indexes[1:])):
            idx_start, idx_stop = start_stop
            if debug:
                print(
                    f"Часть: {idx + 1}",
                    (idx_start, idx_stop),
                    "длительность:",
                    idx_stop - idx_start,
                )
            part_sound = sound[idx_start:idx_stop]

            # длительность меньше порога - не будем предсказывать
            part_len = len(part_sound)
            if part_len < self.audio_min_duration:
                if debug:
                    print(
                        f"Обнаружен фрагмент с длительностью {part_len} менее допустимой"
                    )
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
                    truncation=True,
                )
                # в этом месте на 2-х файлах была ошибка - нужно исследовать
                try:
                    # Перенести данные на GPU
                    inputs = {key: val.to(self.device) for key, val in inputs.items()}

                    logits = self.model(inputs["input_values"][0]).logits
                    predictions = torch.argmax(logits, dim=-1)
                    # Перенести результаты на CPU
                    pred = predictions.cpu().numpy()[0]
                except:
                    pred = -1
            else:
                pred = 0

            predicted_emotion = self.num2emotion.get(pred)
            if debug:
                print("Предсказана эмоция:", predicted_emotion)
            file_emotions.append(predicted_emotion)

        # ищем моду эмоций
        pred = stat_mode(file_emotions)
        # считаем кол-во "недовольных" частей
        angry = file_emotions.count("angry")
        # если недовольных частей более self.angry_threshold %
        if pred != "angry" and angry / sound_parts >= self.angry_threshold:
            pred = "angry"
        if not all(map(bool, (self.feature_extractor, self.model))):
            print("Внимание!!! feature_extractor и model не заданы!")
        return pred, file_emotions


if __name__ == "__main__":
    audio_obj = AudioManipulation()
