import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    names = ['Category','Age','Sex','ALB','ALP','ALT','AST','BIL','CHE','CHOL','CREA','GGT','PROT'] 
    input_file = 'hcvdatclear.csv'
    df = pd.read_csv(input_file,         # Nome do arquivo com dados
                     names = names)      # Nome das colunas 
    print(df.head())

    # Medidas de Tendência Central
    print("\nMedidas de Tendência Central\n")
    print('Média: ', df['Age'].mean()) # Média
    print('Mediana: ', df['Age'].median()) # Mediana
    print('Ponto Médio: ', (df['Age'].max() + df['Age'].min())/2) # Ponto Médio
    print('Moda: ', df['Age'].mode()) # Moda
    
    # Medidas de Dispersão
    print("\nMedidas de Dispersão\n")
    print('Amplitude: ', df['Age'].max() - df['Age'].min())  # Amplitude
    print('Desvio padrão: ', df['Age'].std()) # Desvio padrão
    print('Variância: ', df['Age'].var()) # Variância
    print('Coeficiente de variação: ', (df['Age'].std()/df['Age'].mean())) # Coeficiente de Variação

    # Medidas de Posição Relativa
    print("\nMedidas de Posição Relativa\n")
    print('Z Score:\n', (df['Age'] - df['Age'].mean())/df['Age'].std()) # Z Score
    print('Quantil (25%): ', df['Age'].quantile(q=0.25)) # Quantil 25%
    print('Quantil (50%): ', df['Age'].quantile(q=0.50)) # Quantil 50%
    print('Quantil (75%): ', df['Age'].quantile(q=0.75)) # Quantil 75%
    
    # Medidas de Associação
    print("\nMedidas de Associação\n")
    print('Covariância:\n', df.cov()) # Covariância
    print('\nCorrelação: ', df.corr()) # Correlação

    #plot do gráfico de quartis Categorias x Idades
    plt.figure(figsize = (13,11))
    sns.set_style("whitegrid")
    ax = sns.boxplot( x = "Category", y ="Age",data = df,
                  hue = "Category",linewidth=2, palette = "Set2")
    plt.title("Boxplot da base de dados de HCV", loc="center", fontsize=18)
    plt.xlabel("Categorias")
    plt.ylabel("Idades")
    plt.show()
  
if __name__ == "__main__":
    main()