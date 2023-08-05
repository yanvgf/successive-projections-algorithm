import numpy as np


def normalize_by_train(train, test):
    """Normaliza dados entre -1 e 1 baseado nos dados de treinamento.
    
    Como não conhecemos os dados de teste a priori na prática e precisamos normalizá-los para
    treinar os modelos, precisamos normalizá-los baseado nos parâmetros de normalização dos
    dados de treinamento.
    
    Input:
            train --> matriz com dados de treinamento \n
            test --> matriz com dados de teste \n
    
    Output:
            (train_norm, test_norm) --> matrizes de treino e teste normalizadas
    """
    
    # Função para normalização baseado nos parâmetros do treinamento
    normalize = lambda x, train: 2*(x - np.min(train, axis=0))/(np.max(train, axis=0) - np.min(train, axis=0)) - 1
    
    return normalize(train, train), normalize(test, train)




def train_elm(x, y, neurons, learning='lstsq'):
    """Treina rede ELM a partir dos dados de entrada

    Input: 
            x --> matriz com os dados de entrada do treinamento \n
            y --> matriz com os dados de saída do treinamento \n
            neurons --> número de neurônios na camada escondida \n
            learning (default='lstsq') --> 'hebb', se deseja-se usar aprendizado hebbiano, ou 'lstsq', se \
                    deseja-se utilizar mínimos quadrados \n

    Output: 
            Z --> matriz de pesos da camada de entrada \n
            H --> matriz de projeção dos dados na camada intermediária \n
            W --> matriz de pesos da camada de saída
    """
    
    # Adiciona coluna do termo de polarização em X
    x_aug = np.c_[x, np.ones(x.shape[0])] 
    
    # Em ELMs, os pesos da camada de entrada são aleatórios (neste caso,
    # segundo uma distribuição uniforme)
    Z = np.random.uniform(-1, 1, [x_aug.shape[1], neurons]) # Matriz de pesos da camada de entrada Z
    
    # H = psi(XZ), sendo psi a função de ativação dos neurônios da camada escondida
    H = np.tanh(x_aug @ Z)
    
    # No aprendizado hebbiano, W = X^tY
    if learning=='hebb':
        W = H.T @ y 
    # Na solução de mínimos quadrados, W = X^+Y, sendo X^+ a pseoduinversa de X
    elif learning=='lstsq':
        W = np.linalg.pinv(H) @ y
    
    return Z, H, W




def test_elm(x, Z, W):
    """Obtém saída de rede ELM

    Input: 
            x --> matriz com os dados de entrada do teste \n
            Z, W --> matrizes de pesos da rede (camada de entrada e de saída) \n
    Output:
            yhat --> saída do modelo
    """
        
    # Adiciona coluna do termo de polarização em X
    x_aug = np.c_[x, np.ones(x.shape[0])] 
    
    # H = psi(XZ), sendo psi a função de ativação dos neurônios da camada escondida
    H = np.tanh(x_aug @ Z)
    
    # y^ = HW
    yhat = H @ W

    return(yhat)


def train_pruned_elm(y, H, SEL, learning='lstsq'):
    """Retreina rede ELM após a poda

    Input: 
            y --> matriz com os dados de saída do treinamento \n
            H --> matriz da camada escondida da rede original \n
            SEL --> vetor com neurônios selecionados \n
            learning (default='lstsq') --> 'hebb', se deseja-se usar aprendizado hebbiano, ou 'lstsq', se \
                    deseja-se utilizar mínimos quadrados \n

    Output: 
            H_pruned --> matriz de projeção dos dados na camada intermediária após poda \n
            W --> matriz de pesos da camada de saída após poda
    """
    
    # Poda neurônios que não foram selecionados
    H_pruned = H[:,SEL]
    
    # Obtém nova matriz W
    if learning=='hebb':
        W_pruned = H_pruned.T @ y
    if learning=='lstsq':
        W_pruned = np.linalg.pinv(H_pruned) @ y
        
    return(H_pruned, W_pruned)
    


def test_pruned_elm(x, Z, W_pruned, SEL):
    """Obtém saída de rede ELM podada

    Input: 
            x --> matriz com os dados de entrada do teste \n
            Z --> matriz de pesos da rede original (camada de entrada) \n
            W_pruned --> matriz de pesos da rede podada (camada de saída) \n
            SEL --> vetor com neurônios selecionados \n
    Output:
            yhat_pruned --> saída do modelo
    """
        
    # Adiciona coluna do termo de polarização em X
    x_aug = np.c_[x, np.ones(x.shape[0])]
        
    # Elimina sinapses de Z referentes aos neurônios podados
    Z_pruned = Z[:,SEL]
    
    # Obtém projeção na camada escondida podada
    H_pruned = np.tanh(x_aug @ Z_pruned)
    
    # Obtém saída do modelo
    yhat_pruned = H_pruned @ W_pruned
    
    return(yhat_pruned)


def get_crosstalk(x, y):
    """Calcula o crosstalk de modelo hebbiano

    Input:
            x --> matriz com dados de entrada do treinamento\n
            y --> matriz com dados de saída do treinamento\n
            
    Output:
            crosstalk_series --> crosstalk para cada amostra de treinamento
    """

    # O crosstalk é o erro do modelo hebbiano, então deve-se calculá-lo como
    # uma série de erros para cada amostra de treinamento
    crosstalk_series = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        crosstalk_i = 0
        for j in range(x.shape[0]):
            if i != j:
                crosstalk_i += (x[i] @ x[j].T) * y[j] 
        crosstalk_series[i] = crosstalk_i
        
    return(crosstalk_series)