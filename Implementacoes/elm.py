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


