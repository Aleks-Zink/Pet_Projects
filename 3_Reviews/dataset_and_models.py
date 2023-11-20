import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from gensim.corpora import Dictionary
from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import Doc2Vec


class ReviewDataset(Dataset):
    """
    Класс, для преобразования текста в матрицу (одна строка - одно слово)
    Число строк фиксировано величиной number_of_rows,
    если число слов в тексте меньше, то оставшиеся строки заполняются нулями,
    если число слов в тексте больше, то берутся number_of_rows первых слов.
    """
    def __init__(
            self,
            data,
            model,
            number_of_rows,
            vector_size,
            labels
    ):
        """
        Инициализация объекта
        :param data: Тексты
        :param model: Модель для преобразования слова в вектор
        :param number_of_rows: Число строк итоговой матрицы
        :param vector_size: Длина вектора
        :param labels: Словарь для числового кодирования типа отзыва
        """

        self.data = data
        self.len = self.data.shape[0]
        self.model = model
        self.number_of_rows = number_of_rows
        self.vector_size = vector_size
        self.labels = labels

    def __len__(self):
        """
        :return: длина текста
        """
        return self.len

    def __getitem__(self, ind):
        """
        Выдаёт переработанный текст по индексу
        :param ind: индекс интересующего текста
        :return: Матрица, полученная из текста, Числовой код типа отзыва
        """
        # получение интересующего текста
        text = self.data['review'].iloc[ind]
        # создаём матрицу из нулей
        x = np.zeros((self.number_of_rows, self.vector_size), dtype=np.float32)

        index_word, index_array = 0, 0
        # строчку за стройчкой заменяем вектором, который кодирует слово
        while (index_array < self.number_of_rows) and (index_word < len(text)):
            try:
                x[index_array] = self.model.wv[text[index_word]]
                index_array += 1
            except KeyError:
                pass
            index_word += 1
        # из получившейся матрицы делаем тензор
        x = transforms.ToTensor()(x)
        # получение кода для данного текста
        y = self.labels[self.data['type'].iloc[ind]]

        return x, y


class CNN(nn.Module):
    """
    Архитектура Свёрточной Нейронной Сети для анализа настроения текста взята из статьи

    Cliche M. BB_twtr at SemEval-2017 Task 4: Twitter Sentiment Analysis with CNNs and LSTMs
    https://arxiv.org/abs/1704.06125
    """
    def __init__(
            self,
            max_height_of_convolutions=5,
            number_of_filters=2,
            vector_size=5,
            hidden_out_features=None,
            p=0
    ):
        """
        Инициализация сети
        :param max_height_of_convolutions: максимальная высота одной свёртки
               max_height_of_convolutions >= 2
               все высоты сверток 2, 3, ..., max_height_of_convolutions
        :param number_of_filters: число свёрток одной высоты
        :param vector_size: ширина входной матрицы
        :param hidden_out_features: число нейронов полносвязного слоя
        :param p: вероятность отключения переменных одного слоя при обучении
        """
        super().__init__()

        # если hidden_out_features не задано, то оно равно половине от всего числа свёрток
        if not hidden_out_features:
            hidden_out_features = int((max_height_of_convolutions - 1) * number_of_filters / 2)

        self.number_of_filters = int(number_of_filters)
        # слой для избегания переобучения
        self.dropout = nn.Dropout(p)

        # создание списка содержащий свёрточные слои различных высот
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=number_of_filters,
                    kernel_size=(i, vector_size)
                )
                for i in range(2, max_height_of_convolutions + 1)
            ]
        )

        # создание скрытого полносвязного слоя
        self.hidden = nn.Linear(
            in_features=self.number_of_filters * (max_height_of_convolutions - 1),
            out_features=hidden_out_features
        )

        # создание выходного полносвязного слоя
        self.output = nn.Linear(
            in_features=hidden_out_features,
            out_features=3
        )

    def forward(self, x):
        """
        Получение "предсказания" для текста
        :param x: тензор из текстов формы (Б, 1, В, Ш)
                  Б - число тензор-текстов в одном батче
                  В - высота одного тензор-текста
                  Ш - ширина одного тензор-текста
        :return: список из трёх чисел, интерпритировать их можно 2мя способами:
        1) индекс наибольшего числа - код "настроения" текста
        2) SoftMax от этих чисел - вероятность принадлежности текста к каждому типу "настроения" текста
        """

        # тензор текста проходит по свёрткам различных высот
        list_x = []
        for layer in self.convs:
            # результат свёрток проходит через функцию активации ReLU
            temp_x = nn.ReLU()(layer(x))
            # берём max от результата работы каждого фильтра свёртки
            list_x.append(torch.max(temp_x, dim=2).values)

        # объединяем все выводы в один вектор
        x = torch.concat(
            [out.reshape(-1, self.number_of_filters) for out in list_x],
            dim=1
        )
        x = self.dropout(x)

        # получившийся вектор пропускаем через оставшиеся полносвязные слои
        x = self.hidden(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        x = self.output(x)

        return x


class LSTM(nn.Module):
    """
    Архитектура Рекурентной Нейронной Сети для анализа настроения текста взята из статьи

    Cliche M. BB_twtr at SemEval-2017 Task 4: Twitter Sentiment Analysis with CNNs and LSTMs
    https://arxiv.org/abs/1704.06125
    """
    def __init__(
            self,
            vector_size=5,
            lstm_out_features=20,
            hidden_out_features=None,
            p=0
    ):
        """
        Инициализация сети
        :param vector_size: ширина входной матрицы
        :param lstm_out_features: число нейронов в LSTM слое
        :param hidden_out_features: число нейронов полносвязного слоя
        :param p: вероятность отключения переменных одного слоя при обучении
        """
        super().__init__()

        # если hidden_out_features не задано, то оно равно lstm_out_features
        if not hidden_out_features:
            hidden_out_features = lstm_out_features

        # слой для избегания переобучения
        self.dropout = nn.Dropout(p)

        # создание двунаправленного LSTM слоя
        self.lstm = nn.LSTM(
            input_size=vector_size,
            hidden_size=lstm_out_features,
            batch_first=True,
            bidirectional=True
        )

        # создание скрытого полносвязного слоя
        self.hidden = nn.Linear(
            in_features=2 * lstm_out_features,
            out_features=hidden_out_features
        )

        # создание выходного полносвязного слоя
        self.output = nn.Linear(
            in_features=hidden_out_features,
            out_features=3
        )

    def forward(self, x):
        """
        Получение "предсказания" для текста
        :param x: тензор из текстов формы (Б, 1, В, Ш)
                  Б - число тензор-текстов в одном батче
                  В - высота одного тензор-текста
                  Ш - ширина одного тензор-текста
        :return: список из трёх чисел, интерпритировать их можно 2мя способами:
        1) индекс наибольшего числа - код "настроения" текста
        2) SoftMax от этих чисел - вероятность принадлежности текста к каждому типу "настроения" текста
        """
        # изменение формы входного тензора
        # (Б, 1, В, Ш) -> (Б, В, Ш)
        x = x.reshape(x.shape[0], *x.shape[-2:])

        # результат работы LSTM слоя
        # в качестве результата берётся финальное скрытое состояние LSTM слоя
        x = self.lstm(x)
        # объединение результатов работы forward слоя и forward слоя
        x = torch.concat([x[1][0][0], x[1][0][1]], dim=1)
        x = self.dropout(x)

        # получившийся вектор пропускаем через оставшиеся полносвязные слои
        x = self.hidden(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        x = self.output(x)

        return x


class LSTM_CNN(nn.Module):
    """
    Архитектура данной сети - объединение двух вышеописанных сетей в одну
    """
    def __init__(
            self,
            vector_size=5,
            lstm_out=4,
            max_height_of_convolutions=5,
            number_of_filters=2,
            hidden_out_features=None,
            p=0
    ):
        """
        Инициализация сети
        :param vector_size: ширина входной матрицы
        :param lstm_out: число нейронов в LSTM слое
        :param max_height_of_convolutions: максимальная высота одной свёртки
               max_height_of_convolutions >= 2
               все высоты сверток 2, 3, ..., max_height_of_convolutions
        :param number_of_filters: число свёрток одной высоты
        :param hidden_out_features: число нейронов полносвязного слоя
        :param p: вероятность отключения переменных одного слоя при обучении
        """
        super().__init__()

        # если hidden_out_features не задано, то оно равно половине от всего числа свёрток
        if not hidden_out_features:
            hidden_out_features = int((max_height_of_convolutions - 1) * number_of_filters / 2)

        # слой для избегания переобучения
        self.dropout = nn.Dropout(p)

        # создание двунаправленного LSTM слоя
        self.lstm = nn.LSTM(
            input_size=vector_size,
            hidden_size=lstm_out,
            batch_first=True,
            bidirectional=True
        )

        # создание вышеописанного CNN слоя
        self.cnn = CNN(
            max_height_of_convolutions=max_height_of_convolutions,
            number_of_filters=number_of_filters,
            vector_size=2 * lstm_out,
            hidden_out_features=hidden_out_features,
            p=p
        )

    def forward(self, x):
        """
        Получение "предсказания" для текста
        :param x: тензор из текстов формы (Б, 1, В, Ш)
                  Б - число тензор-текстов в одном батче
                  В - высота одного тензор-текста
                  Ш - ширина одного тензор-текста
        :return: список из трёх чисел, интерпритировать их можно 2мя способами:
        1) индекс наибольшего числа - код "настроения" текста
        2) SoftMax от этих чисел - вероятность принадлежности текста к каждому типу "настроения" текста
        """
        # изменение формы входного тензора
        # (Б, 1, В, Ш) -> (Б, В, Ш)
        x = x.reshape(x.shape[0], *x.shape[-2:])

        # результат работы LSTM слоя
        # в качестве результата берутся все скрытые состояния LSTM слоя
        x = self.lstm(x)[0]
        # изменение формы тензора
        # (Б, В, Ш) -> (Б, 1, В, Ш)
        x = x.reshape(x.shape[0], 1, *x.shape[-2:])
        x = self.dropout(x)

        # получившийся тензор пропускается через CNN слой
        x = self.cnn(x)

        return x


class ENSEMBLE(nn.Module):
    """
    В статье предлагалось объединить несколько CNN и LSTM моделей в ансамбль, для повышения точности.
    Данный класс является надстройкой над вышеописанными обученными моделями для получения взвешенного результата.

    Cliche M. BB_twtr at SemEval-2017 Task 4: Twitter Sentiment Analysis with CNNs and LSTMs
    https://arxiv.org/abs/1704.06125
    """
    def __init__(
            self,
            name=None,
    ):
        """
        Инициализация сети
        :param name: название предобученной сохранённой сети
        """
        super().__init__()

        # загрузка обученных моделей
        self.models = nn.ModuleList(
            [
                torch.load(f'models/{name}_{i+1}.pt')
                for i in range(10)
            ]
        )

        # заморозка параметров обученных моделей
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        # инициализация скрытого полносвязного слоя
        self.hidden = nn.Linear(
            in_features=30, 
            out_features=10
        )

        # инициализация выходного полносвязного слоя
        self.output = nn.Linear(
            in_features=10,
            out_features=3
        )

    def forward(self, x):
        """
        Получение "предсказания" для текста
        :param x: тензор из текстов формы (Б, 1, В, Ш)
                  Б - число тензор-текстов в одном батче
                  В - высота одного тензор-текста
                  Ш - ширина одного тензор-текста
        :return: список из трёх чисел, интерпритировать их можно 2мя способами:
        1) индекс наибольшего числа - код "настроения" текста
        2) SoftMax от этих чисел - вероятность принадлежности текста к каждому типу "настроения" текста
        """
        # анализ текста каждой моделью в ансамбле
        list_x = []
        for model in self.models:
            list_x.append(model(x))

        # объединение ответов всех моделей ансамбля в один вектор
        x = torch.concat(
            [out for out in list_x],
            dim=1
        )

        # пропускание полученного выхода через полносвязные слои
        x = self.hidden(x)
        x = nn.ReLU()(x)
        x = self.output(x)

        return x


# загрузка данных
dictionary = Dictionary.load('data/dictionary')

w2v = Word2Vec.load('data/word2vec.model')

fasttext = FastText.load('data/fasttext.model')

d2v = Doc2Vec.load('data/doc2vec.model')

df_train = pd.read_pickle('data/train/df.pkl')
df_test = pd.read_pickle('data/test/df.pkl')

bow_train = sparse.load_npz('data/train/bow.npz')
bow_test = sparse.load_npz('data/test/bow.npz')

tf_idf_train = sparse.load_npz('data/train/tf_idf.npz')
tf_idf_test = sparse.load_npz('data/test/tf_idf.npz')

w2v_train = np.load('data/train/w2v.npy')
w2v_test = np.load('data/test/w2v.npy')

fasttext_train = np.load('data/train/fasttext.npy')
fasttext_test = np.load('data/test/fasttext.npy')

d2v_train = np.load('data/train/d2v.npy')
d2v_test = np.load('data/test/d2v.npy')


# Глобальные переменные

# В 2_nlp.ipynb было показано, что 99% всех текстов в обучающей выборке состоят из <= 451 слов
# округлив это число получаем 450
NUMBER_OF_WORDS = 450
# Размер вектора одного слова
VECTOR_SIZE = w2v.vector_size
# Словарь для числового кодирования типа отзыва
LABELS = {'bad': 0, 'neutral': 1, 'good': 2}


# Получение батчей тензоров из текстов

# для Word2Vec обработки слов
w2v_data_train = ReviewDataset(
    data=df_train,
    model=w2v,
    number_of_rows=NUMBER_OF_WORDS,
    vector_size=VECTOR_SIZE,
    labels=LABELS
)
w2v_data_train = DataLoader(
    dataset=w2v_data_train, 
    batch_size=100, 
    shuffle=True
)

w2v_data_test = ReviewDataset(
    data=df_test,
    model=w2v,
    number_of_rows=NUMBER_OF_WORDS,
    vector_size=VECTOR_SIZE,
    labels=LABELS
)
w2v_data_test = DataLoader(
    dataset=w2v_data_test, 
    batch_size=100,
    shuffle=True
)

# для FastText обработки слов
fasttext_data_train = ReviewDataset(
    data=df_train,
    model=fasttext,
    number_of_rows=NUMBER_OF_WORDS,
    vector_size=VECTOR_SIZE,
    labels=LABELS
)
fasttext_data_train = DataLoader(
    dataset=fasttext_data_train, 
    batch_size=100, 
    shuffle=True
)

fasttext_data_test = ReviewDataset(
    data=df_test,
    model=fasttext,
    number_of_rows=NUMBER_OF_WORDS,
    vector_size=VECTOR_SIZE,
    labels=LABELS
)
fasttext_data_test = DataLoader(
    dataset=fasttext_data_test, 
    batch_size=100, 
    shuffle=True
)
