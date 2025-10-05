INTRODUCTION 
L'objectif de ce mini projet est de développer une chaîne complète pour résoudre un problème de classification supervisée visant à prédire les prix des maisons en retard 
Nous serons amené à nettoyer et transformer un jeu de données, entraîner plusieurs modèles intelligents, et déployer le modèle final dans un environnement de production à l'aide de technologies modernes telle que Docker, Jenkins, Kubernetes et Kubeflow

Partie 1 : Prétraitement des Données
Étape 1 : Exploration initiale
1. Affichez les cinq premières lignes des ensembles d’entraînement (TR) et de test (TS).

<img width="1028" height="1064" alt="image" src="https://github.com/user-attachments/assets/65eeae9e-9693-4efe-ad21-2a7078a9e96b" />

<img width="1027" height="347" alt="image" src="https://github.com/user-attachments/assets/4138cea7-cd18-4054-ab65-2e14daff9f37" />

Étape 2 : Gestion des valeurs manquantes

. Identifiez les valeurs manquantes dans chaque colonne et calculez leur pourcentage.
Interpréter les résultats.
<img width="984" height="669" alt="image" src="https://github.com/user-attachments/assets/c379ef89-e615-4685-ad3c-0520a5f5b9fb" />

Variables avec un taux élevé de valeurs manquantes (≥ 80%)
  PoolQC (99.52%) : Qualité de la piscine. La quasi-totalité des maisons n'a probablement pas de piscine. Il peut être pertinent de remplacer les valeurs manquantes par "None" ou un indicateur de l'absence de piscine.
  MiscFeature (96.30%) : Présence d'une caractéristique spéciale (par exemple, une cabane de jardin). Beaucoup de maisons ne semblent pas en avoir, donc remplacer les valeurs manquantes par "None" est une option.
  Alley (93.77%) : Accès par une allée. La majorité des maisons n’ont probablement pas d'accès à une allée, donc "None" peut être une bonne solution.
  Fence (80.75%) : Présence d'une clôture. Beaucoup de maisons n’ont pas de clôture, donc "None" est une valeur de remplacement logique.
👉 Interprétation 1 : Ces variables sont souvent absentes car la caractéristique n’existe pas dans la majorité des maisons. Remplacer les valeurs par une catégorie "None" ou "Pas de X" est plus pertinent qu’une imputation par la moyenne ou la médiane

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


<img width="1026" height="669" alt="image" src="https://github.com/user-attachments/assets/77c0ae02-10e0-46f5-a16c-02cc57339d6c" />

3.	 Proposez des solutions pour traiter les valeurs manquantes.
Le dataset présente une large gamme de caractéristiques pour les maisons, avec des variations significatives dans la taille des terrains, l'année de construction et la qualité globale des logements. La superficie des terrains varie fortement, allant de 1 300 à plus de 215 000 pieds carrés, avec une médiane autour de 9 478 pieds carrés. La qualité des maisons, mesurée par OverallQual, oscille entre 1 et 10, avec une moyenne de 6, indiquant que la majorité des logements ont une qualité moyenne à bonne. De plus, la colonne LotFrontage a des valeurs manquantes, ce qui pourrait nécessiter une imputation, idéalement en fonction du quartier.
En ce qui concerne les éléments extérieurs et annexes, la majorité des maisons n’ont ni piscine (PoolArea, médiane = 0), ni porches fermés (EnclosedPorch), ni structures spéciales (MiscVal). La variabilité des prix de vente est importante, avec un prix médian de 163 000 dollars, mais pouvant aller jusqu’à 755 000 dollars. La distribution des prix montre une asymétrie vers les valeurs élevées, suggérant la présence de quelques propriétés de luxe. Enfin, les maisons récentes (années 2000+) semblent mieux représentées, ce qui pourrait influencer les tendances du prix de vente et nécessiter une
.....
  	 .
  	 ..
  	 ....
  	 ......
Partie applicative :
●	Automatisez l’entraînement avec Jenkins pour permettre une exécution continue (CI/CD)lorsque le code ou les données changent.

<img width="1026" height="607" alt="image" src="https://github.com/user-attachments/assets/d22037cb-a97e-4d09-a011-daaa0ba8d461" />


●	Créez un pipeline Jenkins qui :
1.	Télécharge les donnees sur git .
2.	Nettoie et Entraîne les données avec le script Dockerisé.
3.	Entraîne le modèle et stocke les artefacts générés (modèle entraîné, logs, métriques).

<img width="1025" height="247" alt="image" src="https://github.com/user-attachments/assets/9111b53e-c4f0-4107-912f-3621c2645e4e" />
<img width="1025" height="413" alt="image" src="https://github.com/user-attachments/assets/743ac073-c4fa-44c2-a9d4-d15af11deb22" />





