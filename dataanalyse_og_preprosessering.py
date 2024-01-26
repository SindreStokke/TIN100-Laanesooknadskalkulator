# Importering av pakker
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import heatmap
import missingno as msno

def data_analysis():
    # Importering og transformasjon av data
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    train.set_index("Loan_ID", inplace=True)
    test.set_index("Loan_ID", inplace=True)

    # Imputer for manglende verdier på treningssettet
    imputer_train = SimpleImputer(strategy="most_frequent")
    clean_train = pd.DataFrame(imputer_train.fit_transform(train), columns=train.columns)

    # Imputer for manglende verdier på testsettet
    imputer_test = SimpleImputer(strategy="most_frequent")
    clean_test = pd.DataFrame(imputer_test.fit_transform(test), columns=test.columns)

    approval_mapping = {'Y': 1, 'N': 0}
    clean_train['Loan_Status'] = clean_train['Loan_Status'].map(approval_mapping)

    # Konvertér trenigssettets kategoriske kolonner til numeriske binære kolonner ved one-hot-encoding
    train_encoded = pd.get_dummies(clean_train, columns=["Gender", "Married", "Dependents", "Education", "Self_Employed",
                                                           "Property_Area"],drop_first=True)

    # Konvertér testsettets kategoriske kolonner til numeriske binære kolonner ved one-hot-encoding
    test_encoded = pd.get_dummies(clean_test, columns=["Gender", "Married", "Dependents", "Education", "Self_Employed",
                                                           "Property_Area"], drop_first=True)


    train_encoded["Credit_History"] = train_encoded["Credit_History"].astype('uint8')
    test_encoded["Credit_History"] = test_encoded["Credit_History"].astype('uint8')

    # Separerer target-kolonnen fra treningssettet
    x_train = train_encoded.drop(train_encoded.columns[-10], axis=1)
    y_train = train_encoded.iloc[:, -10]


    # Deler opp treningssettet i et treningssett og et valideringssett
    X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
    
    return train, test, train_encoded, test_encoded, x_train, y_train, X_train, X_val, y_val

def plot_correlation_matrix(df):
    df_numerical = df.select_dtypes(exclude="object")
    cols = df_numerical.columns.to_list()
    
    cm = np.corrcoef(df_numerical.values.T)
    (fig,ax) = heatmap(cm,
                row_names=cols,
                column_names=cols,
                figsize=(8,8))
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5) # Dette er nødvendig I matplotlib 3.1.1 for å kompensere for en bug
    plt.show()

def plot_head_table(train, train_encoded):
    fig, ax = plt.subplots(figsize=(10, 4))

    # Oppretter et tabell-plot med justert font-størrelse og skalering
    table = ax.table(cellText=train_encoded.head().values, colLabels=train_encoded.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(2, 1.5)

    # Skjul akser
    ax.axis('off')

    # Lagre plottet som et bilde
    plt.savefig('dataframe_head_dummy.png', bbox_inches='tight', pad_inches=0.2)
    plt.show()


    # Øker plottets størrelse
    plt.figure()
    fig, ax = plt.subplots(figsize=(10, 4))

    # Oppretter et tabell-plot med justert font-størrelse og skalering
    table = ax.table(cellText=train.head().values, colLabels=train.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1.5, 2)

    # Skjul akser
    ax.axis('off')

    # Lagre plottet som et bilde
    plt.savefig('dataframe_head.png', bbox_inches='tight', pad_inches=0.2)
    plt.show()

if __name__ == "__main__":
    # Henter variabler fra funksjonen data_analysis()
    train, test, train_encoded, test_encoded, x_train, y_train, X_train, X_val, y_val = data_analysis()
    
    total_rows_with_nan = train.isna().sum(axis=1).sum()
    print(total_rows_with_nan)
    
    msno.matrix(train)
    
    train_encoded.info()
    train_encoded.head(3)
    
    plot_correlation_matrix(train_encoded)
    
    sns.pairplot(data=train_encoded, hue="Loan_Status")
    
    plot_head_table(train, train_encoded)
