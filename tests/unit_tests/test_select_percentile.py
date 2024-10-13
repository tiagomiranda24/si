from unittest import TestCase
from datasets import DATASETS_PATH
import os
from si.feature_selection.select_percentile import SelectPercentile
from si.io.csv_file import read_csv
from si.statistics.f_classification import f_classification


class TestSelectPercentile(TestCase):

    def setUp(self):
        # Define o caminho para o arquivo CSV do dataset Iris
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        
        # Carrega o dataset a partir do arquivo CSV
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        # Cria uma instância do SelectPercentile com 50% das melhores características
        select_percentile = SelectPercentile(score_func=f_classification, percentile=50)

        # Ajusta o modelo ao dataset
        select_percentile.fit(self.dataset)

        # Verifica se os scores F e p-values foram calculados
        self.assertTrue(select_percentile.F.shape[0] > 0)
        self.assertTrue(select_percentile.p.shape[0] > 0)

    def test_transform(self):
        # Cria uma instância do SelectPercentile com 50% das melhores características
        select_percentile = SelectPercentile(score_func=f_classification, percentile=50)

        # Ajusta o modelo ao dataset
        select_percentile.fit(self.dataset)

        # Transforma o dataset original usando o SelectPercentile
        new_dataset = select_percentile.transform(self.dataset)

        # Verifica se o número de características foi reduzido
        self.assertLess(len(new_dataset.features), len(self.dataset.features))
        self.assertLess(new_dataset.X.shape[1], self.dataset.X.shape[1])

    def test_percentile_value(self):
        # Testa se um percentual inválido gera um erro
        with self.assertRaises(ValueError):
            SelectPercentile(score_func=f_classification, percentile=110)  # Valor fora do intervalo


if __name__ == '__main__':
    import unittest
    unittest.main()
