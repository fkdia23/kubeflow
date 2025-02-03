#!/usr/bin/env python
# coding: utf-8

# Étape 1 : Exploration initiale
# 
# 1. Affichez les cinq premières lignes des ensembles d’entraînement (TR) et de test (TS).
# 	
# 2. Obtenez les dimensions des deux ensembles.
# 
# 3. Identifiez les types de variables présentes dans les datasets.

# In[4]:


import pandas as pd

# Charger les datasets
train_df = pd.read_csv("datasets/train.csv")  # Remplace par le chemin de ton fichier
test_df = pd.read_csv("datasets/test.csv")    # Remplace par le chemin de ton fichier

# 1. Afficher les cinq premières lignes
print("Train dataset (5 premières lignes) :")
print(train_df.head())

# print("\nTest dataset (5 premières lignes) :")
# print(test_df.head())

# 2. Obtenir les dimensions des datasets
print("\nDimensions du dataset d'entraînement :", train_df.shape)
print("Dimensions du dataset de test :", test_df.shape)

# 3. Identifier les types de variables
print("\nTypes de variables du dataset d'entraînement :")
print(train_df.dtypes.value_counts())  # Affiche le nombre de variables de chaque type

print("\nAperçu détaillé des types de données :")
print(train_df.dtypes)


# Si tes datasets sont volumineux, tu peux aussi utiliser .info() pour un aperçu plus rapide :

# In[20]:


train_df.info()


# In[11]:


# test_df.info()


# 1. Identifier les valeurs manquantes et calculer leur pourcentage

# In[18]:


# Identifier les valeurs manquantes et calculer leur pourcentage
missing_train = train_df.isnull().sum()
missing_percent_train = (missing_train / len(train_df)) * 100

missing_test = test_df.isnull().sum()
missing_percent_test = (missing_test / len(test_df)) * 100

# Afficher les colonnes avec des valeurs manquantes et leur pourcentage
print("Valeurs manquantes dans le dataset d'entraînement (en %) :")
print(missing_percent_train[missing_percent_train > 0].sort_values(ascending=False))

# print("\nValeurs manquantes dans le dataset de test (en %) :")
# print(missing_percent_test[missing_percent_test > 0].sort_values(ascending=False))


# 2. Afficher des tableaux de statistiques descriptives

# In[17]:


# Statistiques descriptives pour le dataset d'entraînement
print("\nStatistiques descriptives du dataset d'entraînement :")
print(train_df.describe())

# # Statistiques descriptives pour le dataset de test
# print("\nStatistiques descriptives du dataset de test :")
# print(test_df.describe())


# Voici une analyse et des propositions pour traiter les valeurs manquantes dans ton jeu de données :
# 
# ### 1. **Analyse des valeurs manquantes :**
# 
# #### Colonnes avec un pourcentage élevé de valeurs manquantes :
# - **PoolQC (99.5 %) :** La colonne contient des informations sur la qualité de la piscine, mais la majorité des maisons n'en ont pas, ce qui explique la grande proportion de valeurs manquantes. 
# - **MiscFeature (96.3 %) :** Il s'agit d'une colonne liée à des caractéristiques spéciales comme les cabanons ou autres caractéristiques inhabituelles de la propriété, mais peu de maisons possèdent ces caractéristiques.
# - **Alley (93.8 %) :** Certaines maisons n'ont pas d'allée, ce qui peut expliquer l'absence de données.
# - **Fence (80.8 %) :** La présence de clôtures n'est pas universelle pour toutes les propriétés, d'où le grand nombre de valeurs manquantes.
# - **MasVnrType (59.7 %) :** La colonne indique le type de revêtement extérieur en maçonnerie, mais il se peut que certaines maisons n'en aient pas.
# - **FireplaceQu (47.3 %) :** Comme pour les autres colonnes liées aux caractéristiques spécifiques des maisons (comme la cheminée), une grande proportion de maisons n'en ont pas.
#   
# #### Colonnes avec des valeurs manquantes modérées :
# - **LotFrontage (17.7 %) :** La longueur de la rue peut être manquante pour certaines propriétés, mais ce n'est pas une proportion excessive.
# - **GarageType, GarageYrBlt, GarageFinish, GarageQual, GarageCond (environ 5.5 %) :** Ces colonnes sont liées au garage, et des maisons sans garage auront des valeurs manquantes dans ces colonnes.
# - **Bsmt (ex. BsmtQual, BsmtCond, etc.) (environ 2.5 %) :** Si une maison n'a pas de sous-sol, ces colonnes seront manquantes.
# 
# ### 2. **Solutions pour traiter les valeurs manquantes :**
# 
# #### a) **Colonnes avec des valeurs manquantes massives (> 50 % de valeurs manquantes) :**
# - **PoolQC, MiscFeature, Alley, Fence :** Ces colonnes contiennent des informations qui ne s'appliquent pas à la majorité des maisons. On peut :
#   - Les **supprimer**, car elles ne sont pas essentielles et peuvent fausser les analyses.
#   - **Remplacer par une catégorie "Non applicable"** (par exemple, "No Pool" pour PoolQC) si tu souhaites conserver ces colonnes.
# 
# #### b) **Colonnes avec des valeurs manquantes modérées (entre 10 et 50 %) :**
# - **MasVnrType et FireplaceQu :** Comme ces colonnes indiquent des caractéristiques qui ne sont pas présentes dans toutes les maisons, tu peux :
#   - **Imputer les valeurs manquantes** par la catégorie la plus fréquente, par exemple "None" pour MasVnrType et "No Fireplace" pour FireplaceQu.
#   
# #### c) **Colonnes avec des valeurs manquantes relativement faibles (< 10 %) :**
# - **LotFrontage :** L'imputation par la **moyenne** ou la **médiane** peut être une option ici, étant donné que la proportion de valeurs manquantes n'est pas très élevée. Cependant, il peut être pertinent d'examiner s'il existe des corrélations avec d'autres variables, comme la taille du lot.
# - **GarageType, GarageYrBlt, GarageFinish, GarageQual, GarageCond :** Imputer ces colonnes avec la catégorie la plus fréquente ou avec "No Garage" pourrait être une bonne approche pour les maisons sans garage.
# 
# #### d) **Colonnes liées aux sous-sols (Bsmt) :**
# - **BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2 :** Si un sous-sol est absent, ces informations seront manquantes. Tu peux :
#   - **Imputer par "No Basement"** pour les maisons sans sous-sol.
#   
# #### e) **Cas spécifiques :**
# - **MasVnrArea :** Cette colonne semble être une valeur numérique qui est manquante pour certaines maisons sans revêtement en maçonnerie. L'imputation peut être faite par **0** pour les maisons sans revêtement en maçonnerie ou par la **médiane** des valeurs disponibles.
#   
# #### f) **Méthode avancée :**
# - Si tu souhaites améliorer la précision de l'imputation, tu peux utiliser des techniques comme **l'imputation par régression** ou **l'imputation multiple**, qui utilisent les autres caractéristiques pour estimer les valeurs manquantes.
# 
# ### 3. **Code pour imputer les valeurs manquantes :*
# 

# In[16]:


# Suppression des colonnes avec des valeurs manquantes massives
train_df.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence'], inplace=True)

# Imputation des colonnes avec des catégories manquantes (MasVnrType, FireplaceQu)
train_df['MasVnrType'].fillna('None', inplace=True)
train_df['FireplaceQu'].fillna('No Fireplace', inplace=True)

# Imputation de LotFrontage par la médiane
train_df['LotFrontage'].fillna(train_df['LotFrontage'].median(), inplace=True)

# Imputation des colonnes Garage par la catégorie "No Garage"
garage_columns = ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in garage_columns:
    train_df[col].fillna('No Garage', inplace=True)

# Imputation des colonnes Bsmt par "No Basement"
bsmt_columns = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
for col in bsmt_columns:
    train_df[col].fillna('No Basement', inplace=True)

# Imputation de MasVnrArea par 0 ou la médiane
train_df['MasVnrArea'].fillna(0, inplace=True)


# In[22]:


# Afficher les statistiques descriptives après le traitement
# print(train_df.describe())
train_df.info()


# In[ ]:




