import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    names = ['Category','Age','Sex','ALB','ALP','ALT','AST','BIL','CHE','CHOL','CREA','GGT','PROT'] 
    input_file = 'hcvdatclear.csv'
    df = pd.read_csv(input_file,         # Nome do arquivo com dados
                     names = names)      # Nome das colunas 

    donor = []
    suspect = []
    hepatits = []
    fibrosis = []
    cirrhosis = []

    categorias = [0,0.5,1,2,3]
    for j, i in enumerate(df.iloc[:,0]):
        if i == categorias[0]:
            donor.append(df.iloc[j,1])
        if i == categorias[1]:
            suspect.append(df.iloc[j,1])
        if i == categorias[2]:
            hepatits.append(df.iloc[j,1])
        if i == categorias[3]:
            fibrosis.append(df.iloc[j,1])
        if i == categorias[4]:
            cirrhosis.append(df.iloc[j,1])
            dados_sepal_length = [donor, suspect, hepatits, fibrosis, cirrhosis]

    plt.figure(figsize = (13,11))

    sns.set_style("whitegrid")

    ax = sns.boxplot( x = "Category", y ="Age",data = df,
                  hue = "Category",linewidth=3, palette = "Set3")

    plt.title("Boxplot da base de dados de HCV", loc="center", fontsize=18)
    plt.xlabel("Categorias")
    plt.ylabel("Idades")

    plt.show()

if __name__ == "__main__":
    main()
    
    # bplots = plt.boxplot(dados_sepal_length,  vert = 1, patch_artist = False)

    # colors = ['pink', 'lightblue', 'lightgreen']
    # c = 0
    # for i, bplot in enumerate(bplots['boxes']):
    #     bplot.set(color=colors[c], linewidth=3)
    # c += 1

    # # Linhas de contorno
    # colorss = ['pink','pink', 'lightblue', 'lightblue', 'lightgreen', 'lightgreen']
    # c2 = 0
    # for whisker in bplots['whiskers']:
    #     whisker.set(color=colorss[c2], linewidth=3)
    # c2+= 1

    # c3 = 0
    # for cap in bplots['caps']:
    #     cap.set(color=colorss[c3], linewidth=3)
    # c3 +=1

    # c4 = 0
    # for median in bplots['medians']:
    #     median.set(color=colors[c4], linewidth=3)
    # c4 +=1

    # Adicionando Título ao gráfico