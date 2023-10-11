# Versões das bibliotecas usadas estão listadas no arquivo requirements.txt
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr
from tqdm import tqdm


def projections_qr(X, k, M):
    """Projeções sucessivas das colunas de X em subespaços ortogonais, partindo da coluna
    k e selecionando M colunas ao fim do processo. Referente à Fase 1 do APS.
    
    Essa implementação utiliza a decomposição QR com pivotação de colunas como um "atalho",
    pois é realizado o mesmo processo de busca do APS, mas de forma otimizada. 
    
    Input:
            X --> matriz numpy com dados de calibração \n
            k --> índice da coluna a ser utilizada como vetor de referência inicial \n
            M --> número máximo de variáveis a ser selecionadas \n
            
    Output:
            SEL --> vetor com índice das colunas referentes às variáveis selecionadas            
    """
    
    # Nessa implementação, não é necessário armazenar o X projetado. Apenas
    # escala-se a coluna referente a k para que ela seja o primeiro vetor de 
    # referência na decomposição QR
    X_scaled = X.copy()
    
    # Determina a norma euclidiana de cada coluna de X
    norms = np.sum((X ** 2), axis=0)
    
    # Determina a norma máxima entre as colunas
    norm_max = np.amax(norms)
    
    # Por padrão, a decomposição QR inicia com a coluna de maior norma. Por
    # isso, faz-se a coluna referente a k ter norma duas vezes maior que a 
    # coluna com maior norma para ela ser o primeiro vetor de referência
    X_scaled[:, k] = X_scaled[:, k] * 2 * norm_max / norms[k]
    
    # Na decomposição QR com pivotação de colunas, o vetor de referência em
    # cada iteração é escolhido como aquele com maior norma euclidiana, da
    # mesma forma que o APS é definido.
    #
    # A função qr() do scipy.linalg, quando utilizada com pivotação, retorna
    # como terceiro output o vetor de permutações P, que armazena os índices
    # das variáveis utilizadas como referência em cada iteração. Desse modo,
    # as M primeiras variáveis em P são exatamente as mesmas que seriam
    # selecionadas pelo APS
    _, __, reference_variables = qr(X_scaled, 0, pivoting=True)
    SEL = reference_variables[:M].T
    return SEL




def standardize_data(X):
    """Padroniza a distribuição dos dados da matriz de calibração, que é assumida como normal. Isso permite que apenas
    o ângulo entre os vetores seja determinante para a seleção de variáveis.

    Inputs:
            X --> matriz numpy com os dados de calibração

    Outputs: 
            X_standardized --> matriz numpy com os dados padronizados (média 0 e desvio-padrão 1)
    """

    X_standardized = X.copy()

    # Corrige problema quando desvio-padrão de coluna é 0
    cols_std = np.where(np.std(X, axis=0)==0, 1, np.std(X, axis=0))
    
    # Realiza-se a padronização em cada coluna de X (subtrai a média e divide pelo desvio-padrão)
    X_standardized = (X - np.mean(X, axis=0)) / cols_std
    
    return(X_standardized)




def validate(xcal, ycal, SEL, xval=None, yval=None):
    """Avalia qualidade do modelo obtido com o conjunto de variáveis selecionadas. 
    
    Inputs:
            xcal, ycal --> matrizes numpy com dados para calibração \n
            xval, yval (default=None) --> matrizes numpy com dados para validação. Se não especificadas,
                                        será utilizada a validação cruzada por LOO \n
            SEL --> índices das colunas de X selecionadas na Fase 1 do algoritmo \n
            
    Outputs: 
            RMSE --> erro do modelo
    """
    
    # Número de amostras de calibração
    N = xcal.shape[0]  
    
    # Determina se deve ser usado o conjunto de validação ou LOO
    if xval is None:  
        N_val = 0
    else:
        N_val = xval.shape[0]
    
    # Se há conjunto de validação especificado, usa ele para a validação
    if N_val > 0:        
        # Regressão linear múltipla
        xcal_augmented = np.hstack([np.ones((N, 1)), xcal[:, SEL].reshape(N, -1)])
        w = np.linalg.lstsq(xcal_augmented, ycal, rcond=None)[0]
        # Previsões no conjunto de validação
        xval_augmented = np.hstack([np.ones((N_val, 1)), xval[:, SEL].reshape(N_val, -1)])
        yhat = xval_augmented.dot(w)
        
        # Cálculo do RMSE
        RMSE = np.sqrt(np.mean((yval-yhat)**2, axis=0))
        
    # Se não há conjunto de validação, usa-se LOO no conjunto de calibração    
    else:
        # Define o tamanho adequado para yhat
        yhat = np.zeros((N, 1))
        
        # A cada iteração, seleciona-se uma amostra do conjunto de calibração para servir de
        # validação e constrói-se o modelo com todas as demais amostras (LOOCV - Leave-One-Out Cross-Validation)
        for i in range(N):
            # Remove o item i do conjunto de calibração e usa ele como validação
            cal = np.hstack([np.arange(i), np.arange(i + 1, N)])
            xcal_loo = xcal[cal, :][:, SEL]
            ycal_loo = ycal[cal]
            xval_loo = xcal[i, SEL]
            
            # Regressão linear múltipla
            xcal_loo_augmented = np.hstack([np.ones((N - 1, 1)), xcal_loo.reshape(N - 1, -1)])
            w = np.linalg.lstsq(xcal_loo_augmented, ycal_loo, rcond=None)[0]
                
            # Previsão na amostra de validação
            yhat[i] = np.hstack([np.ones(1), xval_loo]).dot(w)
            
        # Cálculo do RMSE do LOO
        RMSE = np.sqrt(np.mean((ycal-yhat.flatten())**2, axis=0))

    return(RMSE)




def optimize_hyperparameters(SEL_all, xcal, ycal, M_min, M_max, xval=None, yval=None):
    """Obtém, por grid-search, a melhor combinação de hiperparâmetros para o APS.
    
    Input:
            SEL_all --> matriz (M_max x K) que armazena, em cada coluna, o conjunto de variáveis selecionadas
                        pelo APS ao usar cada coluna de X como vetor de referência inicial \n
            xcal, ycal --> matrizes com dados de calibração \n
            M_min, M_max --> número mínimp e máximo de variáveis no conjunto final \n 
            xval, yval (default=None) --> matrizes com dados de validação. Se não forem especificadas, será
                                            realizada validação cruzada por LOO \n
    Output:
            SEL_star --> conjunto de variáveis selecionadas cujos hiperparâmetros (M, k) otimizam o modelo de MLR
    """
    
    K = xcal.shape[1] 
    
    # Cria matriz de RMSEs com (M_max + 1) linhas e K colunas preenchida inicialmente por infinitos. Essa matriz
    # armazenará o RMSE para cada par (M, k) de hiperparâmetros do APS
    RMSE = float('inf') * np.ones((M_max + 1, K))
    
    # Realiza-se grid-search para encontrar combinação de hiperparâmetros que retorna o melhor modelo
    for k in tqdm(range(K), desc="Evaluation of variable subsets"):
        for M in range(M_min, M_max + 1):
            SEL = SEL_all[:M, k].astype(int)
            
            # Avalia modelo construído com o conjunto de variáveis dessa iteração e adiciona na matriz de RMSE
            RMSE[M, k] = validate(xcal, ycal, SEL, xval, yval)
    
    # Os melhores hiperparâmetros são aqueles que minimizam o RMSE
    RMSE_min = np.min(RMSE, axis=0)
    M_star = np.argmin(RMSE, axis=0)
    k_star = np.argmin(RMSE_min)
    
    # SEL_star é a saída do APS quando utilizados os hiperparâmetros ótimos, (M_star, k_star)
    SEL_star = SEL_all[:M_star[k_star], k_star]
    
    return(SEL_star)




def spa(xcal, ycal, xval=None, yval=None, M_min=1, M_max=None, plot=True):
    """Executa seleção de variáveis baseada nas duas primeiras fases do Algoritmo das Projeções Sucessivas
    
    Input:
            xcal, ycal --> matrizes com dados de calibração \n
            xval, yval (default=None) --> matrizes com dados de validação. Se não foram
                                            especificadas, será utilizado validação cruzada
                                            por LOO \n
            M_min (default=1), M_max (default=None) --> número mínimo e máximo de variáveis
                                                        que devem ser selecionadas \n
    
    Output:
            SEL_star --> conjunto final de variáveis selecionadas
    """
    
    N, K = xcal.shape
    
    # Verificação da validade dos parâmetros passados
    if M_max is None:
        if xval is None:
            # O número máximo de variáveis selecionadas é N-1, pois cada projeção tira
            # um grau de liberdade dos vetores
            #
            # Caso não seja definido um conjunto de validação, o número máximo de variáveis
            # selecionadas é N-1, pois uma amostra é reservada para a validação cruzada com LOO
            M_max = min(N-2, K)
        else:
            M_max = min(N-1, K)
    assert (M_max <= min(N-1, K)), "Parâmetro M_max inválido: o número máximo de variáveis selecionadas é N"
    
    # Tratamento dos dados (todas as colunas ficam com média 0 e desvio-padrão 1)
    xcal_standardized = standardize_data(xcal)
    
    # Fase 1: Cálculo das projeções para cada k inicial
    #
    # Cada coluna de SEL_all corresponde ao conjunto de variáveis de tamanho M_max que seria selecionado
    # na fase 1 do APS. Essa matriz será utilizada posteriormente na seleção dos hiperparâmetros que geram
    # o melhor modelo
    SEL_all = np.zeros((M_max, K))
    for k in tqdm(range(K), desc='Projections'):
        SEL_all[:, k] = projections_qr(xcal_standardized, k, M_max)
        
    # Fase 2: Grid-search para obter par (k, M) que produz o melhor modelo
    SEL_star = optimize_hyperparameters(SEL_all, xcal, ycal, M_min, M_max, xval, yval)
    SEL_star = SEL_star.astype(int)
    
    if plot is True:
        # Vou marcar os comprimentos de onda selecionados em cima da primeira amostra de calibração,
        # como os autores originais fazem no GUI
        plt.plot(xcal[0, :])
        plt.scatter(SEL_star, xcal[0, SEL_star], marker='s', color='r')
        plt.legend(['First calibration object', 'Selected variables'],
                fontsize=11, frameon=True, loc='best')
        plt.xlabel('Variable index')
        plt.show()
    
    return(SEL_star)