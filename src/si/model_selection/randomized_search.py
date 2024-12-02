import itertools
from typing import Callable, Tuple, Dict, Any
import numpy as np
from si.data.dataset import Dataset
from si.model_selection.cross_validate import k_fold_cross_validation
from si.models import Model  # Certifique-se de importar a classe ou tipo correto

def randomized_search_cv(
    model: Model, 
    dataset: Dataset, 
    hyperparameter_grid: Dict[str, Tuple], 
    scoring: Callable = None, 
    cv: int = 5, 
    n_iter: int = 10  # Adicionando a variável n_iter
) -> Dict[str, Any]:
    """
    Realiza uma busca aleatória para seleção de hiperparâmetros usando validação cruzada k-fold.
    
    Arguments:
        - model – modelo a ser validado.
        - dataset – dataset para validação.
        - hyperparameter_grid - Dicionário com os nomes dos hiperparâmetros e seus valores para pesquisa.
        - scoring – função para calcular o escore do modelo (opcional, por padrão None).
        - cv – número de folds para validação cruzada (default é 5).
        - n_iter – número de combinações aleatórias de hiperparâmetros a testar.

    Expected output:
        - Um dicionário com os resultados da validação cruzada para cada conjunto de hiperparâmetros.
          Inclui os escores, hiperparâmetros, melhores hiperparâmetros e o melhor escore.
    """
    
    # Lista para armazenar os resultados
    results = []

    # Gerar as combinações aleatórias dos hiperparâmetros
    hyperparameter_combinations = list(itertools.product(*hyperparameter_grid.values()))
    
    # Se n_iter for menor que o número total de combinações, sorteia combinações aleatórias
    if n_iter < len(hyperparameter_combinations):
        sampled_combinations = np.random.choice(len(hyperparameter_combinations), n_iter, replace=False)
        hyperparameter_combinations = [hyperparameter_combinations[i] for i in sampled_combinations]
    else:
        # Caso contrário, usa todas as combinações
        sampled_combinations = range(len(hyperparameter_combinations))

    # Executa validação cruzada k-fold para cada combinação de hiperparâmetros
    for combination in hyperparameter_combinations:
        # Mapear os valores da combinação para os parâmetros do modelo
        hyperparameters = dict(zip(hyperparameter_grid.keys(), combination))
        model.set_params(**hyperparameters)  # Ajustar o modelo com os hiperparâmetros

        # Aplicar validação cruzada k-fold
        scores = k_fold_cross_validation(model, dataset, cv=cv, scoring=scoring)

        # Armazenar os resultados
        results.append({
            'hyperparameters': hyperparameters,
            'scores': scores
        })
    
    # Obter o melhor modelo baseado na pontuação média
    best_result = max(results, key=lambda x: np.mean(x['scores']))
    best_hyperparameters = best_result['hyperparameters']
    best_score = np.mean(best_result['scores'])

    # Retornar os resultados
    return {
        'results': results,
        'best_hyperparameters': best_hyperparameters,
        'best_score': best_score
    }
