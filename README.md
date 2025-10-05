 
 
RAPPORT MINI PROJET MLOPS - DEVOPS - KUBEFLOW
 
Paris 14 Février 2025 
  
 	 
Majeur : Master of Science , Data Engineering 
 
Membres du groupe : 
-	Edson KANOU TAYOUTSTOP 
-	Daina Stela KAMTA TCHOUYON 
-	Franklin KANA NGUEDIA  
 	 	 	 	 	 	 

Table des matières
INTRODUCTION	2
Partie 1 : Prétraitement des Données	2
Étape 1 : Exploration initiale	2
Étape 2 : Gestion des valeurs manquantes	4
Variables avec un taux élevé de valeurs manquantes (≥ 80%)	4
Variables avec un taux modéré de valeurs manquantes (10% 60%)	4
Variables avec peu de valeurs manquantes (≤ 5.5%)	5
Partie applicative 1:	7
Partie 2 : Sélection des Variables Pertinentes	8
Étape 1 :  Calculer la corrélation entre les variables	8
Etape 2 : Comparez les deux approches (corrélation et PCA) et interprétez les résultats	9
Etape 3 : Application du PCA	9
Interprétation et recommandations	9
Partie 3 : Rééchantillonnage des Données Déséquilibrées	10
2.	Afficher la distribution de la valeur de sortie après	11
Partie 4 : Modélisation et Évaluation	12
Étape 1 : Conception des modèles intelligents	12
Comparez les performances des modèles et discutez des résultats.	13
Analyse globale des performances	13
Partie applicative :	14




















INTRODUCTION 
L'objectif de ce mini projet est de développer une chaîne complète pour résoudre un problème de classification supervisée visant à prédire les prix des maisons en retard 
Nous serons amené à nettoyer et transformer un jeu de données, entraîner plusieurs modèles intelligents, et déployer le modèle final dans un environnement de production à l'aide de technologies modernes telle que Docker, Jenkins, Kubernetes et Kubeflow.

Partie 1 : Prétraitement des Données
Étape 1 : Exploration initiale
1. Affichez les cinq premières lignes des ensembles d’entraînement (TR) et de test (TS).
 
 
 
1.	Obtenez les dimensions des deux ensembles.
 
2.	Identifiez les types de variables présentes dans les datasets.
 

Étape 2 : Gestion des valeurs manquantes

1. Identifiez les valeurs manquantes dans chaque colonne et calculez leur pourcentage.
Interpréter les résultats.
 
Variables avec un taux élevé de valeurs manquantes (≥ 80%)
  PoolQC (99.52%) : Qualité de la piscine. La quasi-totalité des maisons n'a probablement pas de piscine. Il peut être pertinent de remplacer les valeurs manquantes par "None" ou un indicateur de l'absence de piscine.
  MiscFeature (96.30%) : Présence d'une caractéristique spéciale (par exemple, une cabane de jardin). Beaucoup de maisons ne semblent pas en avoir, donc remplacer les valeurs manquantes par "None" est une option.
  Alley (93.77%) : Accès par une allée. La majorité des maisons n’ont probablement pas d'accès à une allée, donc "None" peut être une bonne solution.
  Fence (80.75%) : Présence d'une clôture. Beaucoup de maisons n’ont pas de clôture, donc "None" est une valeur de remplacement logique.
👉 Interprétation 1 : Ces variables sont souvent absentes car la caractéristique n’existe pas dans la majorité des maisons. Remplacer les valeurs par une catégorie "None" ou "Pas de X" est plus pertinent qu’une imputation par la moyenne ou la médiane.
Variables avec un taux modéré de valeurs manquantes (10% 60%)
  MasVnrType (59.73%) et MasVnrArea (0.54%) : Type et surface de la maçonnerie en pierre (masonry veneer). Beaucoup de maisons n’ont probablement pas de revêtement en pierre, donc "None" pour le type et 0 pour l'aire sont des imputations possibles.
  FireplaceQu (47.26%) : Qualité de la cheminée. La moitié des maisons n’a pas de cheminée, donc "None" est adapté.
  LotFrontage (17.74%) : Longueur de la façade donnant sur la rue. Les valeurs manquantes pourraient être imputées par la médiane en fonction du quartier
(Neighborhood), car des quartiers similaires ont souvent des largeurs de lots similaires.
👉 Interprétation 2 : Certaines caractéristiques (ex. cheminée, maçonnerie) sont absentes par nature, donc un remplacement par "None" ou 0 est pertinent. Pour LotFrontage, il est plus judicieux d’utiliser des méthodes basées sur des caractéristiques similaires (ex. médiane par quartier).
Variables avec peu de valeurs manquantes (≤ 5.5%)
  GarageType, GarageYrBlt, GarageFinish, GarageQual, GarageCond (5.55%) : Informations sur le garage. Ici, il est probable que les maisons sans garage soient responsables des valeurs manquantes. Il serait pertinent de remplacer :
GarageType par "None"
GarageYrBlt par 0 (ou médiane si pertinent)
GarageFinish , GarageQual , GarageCond par "None"
BsmtExposure, BsmtFinType2, BsmtQual, BsmtCond, BsmtFinType1 (2.5% - 2.6%) : Informations sur le sous-sol. Comme pour le garage, les valeurs manquantes peuvent être remplacées par "None" si elles signifient une absence de sous-sol.
  Electrical (0.07%) : Type de système électrique. Seule une très petite partie des données est affectée. Une imputation par la valeur la plus fréquente (mode) est appropriée ici.
👉 Interprétation 3 : Les garages et sous-sols absents entraînent des valeurs nulles. Remplacer par "None" est logique. Pour Electrical, une imputation par le mode est suffisante.
✅ Catégories à remplacer par "None" : PoolQC, MiscFeature, Alley, Fence, FireplaceQu,
GarageType, GarageFinish, GarageQual, GarageCond, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2.
✅ Catégories à remplacer par 0 : MasVnrArea, GarageYrBlt (si pertinent).
✅ Catégories à imputer par la médiane (en groupe par quartier) : LotFrontage.
✅ Catégories à imputer par la valeur la plus fréquente (mode) : Electrical.
2.Affichez des tableaux de statistiques descriptives pour TR et TS (moyennes, médianes, écarts-types).
 
3.	 Proposez des solutions pour traiter les valeurs manquantes.
Le dataset présente une large gamme de caractéristiques pour les maisons, avec des variations significatives dans la taille des terrains, l'année de construction et la qualité globale des logements. La superficie des terrains varie fortement, allant de 1 300 à plus de 215 000 pieds carrés, avec une médiane autour de 9 478 pieds carrés. La qualité des maisons, mesurée par OverallQual, oscille entre 1 et 10, avec une moyenne de 6, indiquant que la majorité des logements ont une qualité moyenne à bonne. De plus, la colonne LotFrontage a des valeurs manquantes, ce qui pourrait nécessiter une imputation, idéalement en fonction du quartier.
En ce qui concerne les éléments extérieurs et annexes, la majorité des maisons n’ont ni piscine (PoolArea, médiane = 0), ni porches fermés (EnclosedPorch), ni structures spéciales (MiscVal). La variabilité des prix de vente est importante, avec un prix médian de 163 000 dollars, mais pouvant aller jusqu’à 755 000 dollars. La distribution des prix montre une asymétrie vers les valeurs élevées, suggérant la présence de quelques propriétés de luxe. Enfin, les maisons récentes (années 2000+) semblent mieux représentées, ce qui pourrait influencer les tendances du prix de vente et nécessiter une
	





Partie applicative 1:
● Utilisez Docker pour conteneuriser votre script Python de nettoyage des données afin de garantir une exécution reproductible avec les dépendances nécessaires (Pandas,NumPy, etc.).
 









Partie 2 : Sélection des Variables Pertinentes

Étape 1 :  Calculer la corrélation entre les variables 

Appliquer le test de khi deux pour les variables catégorielles. Si la valeur de corrélation dépasse 90% (les deux variables sont fortement corrélées) éliminer l’une de valeurs de DS. Laquelle qu’on doit éliminer et pourquoi ? C’est quoi la nouvelle dimensionnalité de notre DS ?
 
 
Etape 2 : Comparez les deux approches (corrélation et PCA) et interprétez les résultats

Analyse des corrélations

Aucune variable n’a dépassé le seuil de 0.9 en corrélation, ce qui signifie que le dataset ne contient pas de redondances significatives. Par conséquent, la suppression des variables corrélées n’a pas réduit la dimensionnalité, qui reste à  77 variables.

Etape 3 : Application du PCA 

Le PCA a permis de ramener la dimensionnalité de 77 à 27 composantes principales tout en conservant 95% de la variance totale. Cela montre une forte compression de l’information, réduisant la complexité du dataset de plus de 65% sans perte majeure d’information.
Interprétation et recommandations
-	Si l’objectif est la performance d’un modèle prédictif, il est préférable d’utiliser les 27 composantes principales issues du PCA, car elles résument efficacement les données et accélèrent les calculs
-	Si l’interprétabilité est essentielle, il est préférable de conserver toutes les variables d'origine, puisque la corrélation n’a révélé aucune redondance majeure.
Le PCA se révèle donc être la méthode la plus efficace pour ce dataset spécifique, offrant une réduction significative de la dimensionnalité tout en conservant une forte capacité explicative









Partie 3 : Rééchantillonnage des Données Déséquilibrées
1.	Appliquer une méthode de suréchantillonnage (Oversampling) pour résoudre le problème de déséquilibre.
 
 
Interprétation :
La distribution des prix suit une tendance normale pour l’immobilier, avec une grande majorité de prix modérés et quelques valeurs extrêmes.
L’asymétrie pourrait impacter certains modèles, en particulier ceux sensibles aux valeurs extrêmes (comme les régressions linéaires classiques). Des transformations comme log(SalePrice) pourraient améliorer les performances. Le dataset initial était déjà remarquablement équilibré
le réechantillonage n'était pas nécessaire vu le faible déséquilibre initial
 
2.	Afficher la distribution de la valeur de sortie après le rééchantillonnage.
 

Le rééchantillonnage n'était pas nécessaire vu le faible déséquilibre initial












Partie 4 : Modélisation et Évaluation
Étape 1 : Conception des modèles intelligents
 Entraînez un classificateur basé sur un  modèle :
○ Artificial Neural Network (ANN)
 

 
 
Comparez les performances des modèles et discutez des résultats.
Analyse globale des performances
Métrique	Avant échantillonnage	Après échantillonnage	Variation
Accuracy	91%	88%	🔻 -3%
Macro avg F1-score	0.91	0.88	🔻 -0.03
Weighted avg F1-score	0.91	0.88	🔻 -0.03
			
Classe	Faux positifs avant	Faux positifs après	Faux négatifs avant	Faux négatifs après
0	5	3	5	3
1	16	18	16	18
2	2	5	2	5
3	3	7	3	7
📌 Classe 0 : Moins de faux positifs et faux négatifs → amélioration.
📌 Classe 1 : Augmentation des erreurs, notamment en classification avec la classe 0.
📌 Classe 2 : Plus d’erreurs après échantillonnage.
📌 Classe 3 : Plus d’erreurs également.
Observation générale : L’accuracy et le F1-score global ont légèrement baissé après l'échantillonnage. Cela peut être dû à un meilleur équilibrage des classes, qui peut parfois introduire de légères pertes sur certaines performances globales.
L’échantillonnage a amélioré l’équilibre des classes, mais au prix d’une légère baisse de performance globale.

Partie applicative :
●	Automatisez l’entraînement avec Jenkins pour permettre une exécution continue (CI/CD)lorsque le code ou les données changent.
 
 
●	Créez un pipeline Jenkins qui :
1.	Télécharge les donnees sur git .
2.	Nettoie et Entraîne les données avec le script Dockerisé.
3.	Entraîne le modèle et stocke les artefacts générés (modèle entraîné, logs, métriques).
 
 
NB : Le scripts clean_data.ipynb fait office de precessing , et d'entrainement dans notre cas
Partie 5 : Déploiement en Production
Étape 1 : Conteneurisation avec Docker Compose structure globale de l'application
📂 kubeflow/
│── 📂 api/ # Contient l'API (Flask)
│ │── app.py # Code du serveur API
│ │── requirements.txt # Dépendances de l'API
│ │── Dockerfile # Dockerfile pour l'API
│
│── 📂 work/ # Contient le modèle et son serveur
│ │── clean_data.py # Script d'entraînement et pre-traitement
│ │── predict.py # Script d'inférence
│ │── model.pkl # Modèle entraîné
│ │── requirements.txt # Dépendances du modèle
│ │── Dockerfile # Dockerfile pour le modèle
│
│── 📂 dataset/ # Contient les données brutes et nettoyées
│ │── train.csv
│ │── test.csv
|
│── Jenkinsfile # Contient le script du pipeline Jenkins
│
│── 📂 db/ # Contient la configuration de la base de données
│ │── init.sql # Script SQL d'initialisation (facultatif)
│
│── docker-compose.yml # Orchestration avec Docker Compose
│── .gitignore # Fichier Git ignore
│── README.md # Documentation du projet

1. Utilisez Docker Compose pour orchestrer l’environnement, incluant :
○ Un conteneur pour le modèle déployé.
○ Un conteneur pour un serveur API (par exemple, Flask ou FastAPI).
○ Un conteneur pour une base de données de stockage des prédictions.
 
 
 
