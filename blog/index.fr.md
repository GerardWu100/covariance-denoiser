---
title: "Nettoyer une covariance ne suffit pas à prévoir la variance"
description: "Un test walk-forward des features de covariance issues de la théorie des matrices aléatoires et de Ledoit-Wolf sur huit ETF, avec un résultat négatif qui mérite d'être compris."
date: 2026-07-12
image: "images/cover-covariance-denoising.png"
categories: ["Quantitative Finance", "Risk Modeling"]
---

Une matrice de covariance mieux conditionnée devrait rendre un modèle de risque
plus stable. Rien ne garantit que les features tirées de cette matrice prévoiront
mieux la variance. J'ai testé cette distinction sur huit fonds négociés en bourse
(ETF) entre 2008 et 2024. Les features débruitées alimentent une régression ridge,
évaluée strictement hors échantillon. Une prévision naïve fondée sur la dernière
valeur observée fait mieux.

Ce résultat est plus instructif qu'une victoire étroite du modèle. La théorie des
matrices aléatoires (RMT, pour *random matrix theory*) remplit son rôle : elle
réduit le mauvais conditionnement de la covariance. C'est à l'étape suivante que
le raisonnement casse, lorsque la stabilité numérique est supposée devenir de
l'information prédictive.

![Une matrice de covariance bruitée traverse un filtre spectral et ressort sous une forme plus nette.](images/cover-covariance-denoising.png)

L'image reprend les deux étapes distinctes de l'expérience : filtrer la matrice
bruitée à gauche, puis vérifier si la structure obtenue à droite contient de
l'information sur l'avenir.

## Pourquoi nettoyer la covariance

Soit $P_{i,t}$ le cours de clôture ajusté de l'actif $i$ au jour de bourse $t$. Son
rendement logarithmique est

$$
r_{i,t}=\log\left(\frac{P_{i,t}}{P_{i,t-1}}\right).
$$

Plaçons $T$ observations de $N$ actifs dans une matrice de rendements $R$. Si
$\bar R$ est la matrice dont chaque ligne contient les moyennes des colonnes de
$R$, la covariance empirique vaut

$$
S=\frac{1}{T-1}(R-\bar R)^\top(R-\bar R).
$$

Ici, $T=63$ jours de bourse et $N=8$ ETF : EEM, GLD, HYG, IWM, QQQ, SPY, TLT et
VNQ. Soixante-trois observations suffisent pour inverser une matrice de dimension
huit, mais ce critère est peu exigeant. L'erreur d'échantillonnage peut encore
produire de petites valeurs propres instables. Le nombre de conditionnement,
défini comme le rapport entre la plus grande et la plus petite valeur singulière,
mesure cette sensibilité. Lorsqu'il est élevé, une faible variation des données
peut fortement modifier les calculs qui dépendent de la matrice.

J'ai comparé l'estimateur empirique à deux méthodes de nettoyage. Le shrinkage de
Ledoit-Wolf ramène la covariance empirique vers une cible structurée. Le nettoyage
RMT travaille dans l'espace des corrélations. Définissons le ratio observations-
dimension par $q=T/N$. Dans le modèle de Marchenko-Pastur, la borne supérieure du
bulk de bruit est

$$
\lambda_+=\left(1+q^{-1/2}\right)^2.
$$

L'implémentation remplace chaque valeur propre de corrélation $\lambda_i$ telle
que $\lambda_i\leq\lambda_+$ par la moyenne des valeurs propres considérées comme
du bruit. Elle reconstruit ensuite une matrice de corrélation à diagonale unitaire,
puis la remet à l'échelle avec les volatilités empiriques.

```python
eigenvalues, eigenvectors = np.linalg.eigh(sample_correlation.to_numpy())
q = len(returns_window) / returns_window.shape[1]
lambda_plus = (1.0 + q**-0.5) ** 2

noise_mask = eigenvalues <= lambda_plus
eigenvalues[noise_mask] = eigenvalues[noise_mask].mean()
cleaned = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
```

Sur la dernière fenêtre de 63 jours, le nombre de conditionnement de la covariance
empirique atteint 271,2. Ledoit-Wolf le ramène à 9,1 et RMT à 91,1. Les deux
méthodes ne sont pas équivalentes, mais chacune réduit la sensibilité de
l'estimation.

![Nombres de conditionnement des covariances empirique, Ledoit-Wolf et RMT sur la dernière fenêtre de 63 jours.](images/01-condition-numbers.png)

Ledoit-Wolf produit la plus forte réduction sur cette fenêtre. RMT est moins
agressif, mais son nombre de conditionnement ne représente plus qu'environ un
tiers de celui de l'estimation empirique.

## Passer des matrices à une prévision

Le modèle ne prévoit pas directement une matrice de covariance. Pour chaque
fenêtre glissante de 63 jours, il enregistre sept features :

- la corrélation empirique moyenne entre les paires d'actifs ;
- les nombres de conditionnement des covariances empirique, Ledoit-Wolf et RMT ;
- le rapport entre chaque nombre de conditionnement nettoyé et celui de la
  covariance empirique ;
- la volatilité réalisée annualisée sur 21 jours d'un portefeuille équipondéré.

Soit $r_{\mathrm{eq},t}=N^{-1}\sum_{i=1}^{N}r_{i,t}$ le rendement du portefeuille
équipondéré. Pour un horizon de prévision de $h=21$ jours de bourse, la variance
réalisée future annualisée est

$$
RV_{t,h}=\frac{252}{h}\sum_{j=1}^{h}r_{\mathrm{eq},t+j}^{2}.
$$

L'alignement des dates est essentiel. Chaque feature à la date $t$ utilise les
rendements disponibles jusqu'à $t$ inclus. La cible couvre les rendements de
$t+1$ à $t+h$. Aucun rendement futur ne se glisse dans les features.

L'évaluation suit une fenêtre d'apprentissage croissante. Le premier fold compte
252 lignes d'apprentissage et 21 lignes de test. Chaque fold suivant avance de
21 lignes tout en conservant l'historique précédent. On obtient 187 folds et
3 927 prévisions hors échantillon, d'avril 2009 à novembre 2024.

```python
test_start = config.min_train_size
while test_start + config.test_size <= sample_count:
    train_slice = slice(0, test_start)
    test_slice = slice(test_start, test_start + config.test_size)
    slices.append((train_slice, test_slice))
    test_start += config.step_size
```

Le benchmark attribue à chaque observation du fold de test la dernière cible
connue dans l'échantillon d'apprentissage. Le modèle concurrent est une
régression ridge, c'est-à-dire une régression linéaire munie d'une pénalité L2
qui limite les coefficients trop élevés. Le paramètre de pénalisation reste fixé
à 1,0.

## Le benchmark l'emporte

Soit $y_k$ la variance réalisée et $\hat y_k$ sa prévision pour l'observation hors
échantillon $k$, avec $K$ prévisions au total. L'erreur absolue moyenne (MAE) vaut

$$
\operatorname{MAE}=\frac{1}{K}\sum_{k=1}^{K}|y_k-\hat y_k|,
$$

et la racine de l'erreur quadratique moyenne (RMSE) vaut

$$
\operatorname{RMSE}=\sqrt{\frac{1}{K}\sum_{k=1}^{K}(y_k-\hat y_k)^2}.
$$

Les deux mesures sont exprimées en unités de variance annualisée. Une valeur plus
faible est préférable.

| Modèle | MAE | RMSE |
|---|---:|---:|
| Dernière valeur observée | 0.00797 | 0.03134 |
| Régression ridge | 0.01908 | 0.03613 |

La MAE de ridge dépasse celle du benchmark de 139,5 %, et sa RMSE de 15,3 %. La
différence entre ces deux comparaisons donne un indice sur la nature des erreurs.
La mise au carré rend la RMSE particulièrement sensible aux gros écarts. Les deux
modèles souffrent lors des pics de variance ; ridge paraît donc moins mauvais en
RMSE qu'en MAE, sans battre le benchmark pour autant.

![Erreur absolue moyenne et racine de l'erreur quadratique moyenne des deux prévisions hors échantillon.](images/02-forecast-errors.png)

La dernière valeur observée est rudimentaire, mais la persistance de la volatilité
lui donne un avantage difficile à battre. Ridge produit une estimation
conditionnelle plus lisse, tandis que la cible à 21 jours peut changer brutalement
à mesure que les chocs entrent dans sa fenêtre future ou en sortent.

## Ce qui échoue, et ce qui fonctionne

L'estimateur RMT ne rate pas son propre test. Il modifie le spectre des valeurs
propres et réduit le nombre de conditionnement de la covariance. Cela montre une
amélioration du conditionnement, pas de la qualité des prévisions.

Le dispositif demande à sept résumés contemporains de prévoir la variance à
21 jours d'un portefeuille équipondéré. Plusieurs raisons peuvent expliquer le
résultat :

- les nombres de conditionnement décrivent une géométrie numérique, pas
  nécessairement la direction de la volatilité future ;
- une cible à 21 jours dépend fortement de chocs qu'une covariance passée ne peut
  pas anticiper ;
- ridge est linéaire et conserve la même pénalité dans chaque fold ;
- la feature de volatilité réalisée passée peut déjà capter l'essentiel de la
  persistance exploitable ;
- avec huit ETF diversifiés, le nettoyage du bruit en grande dimension dispose
  de peu d'espace pour démontrer son principal avantage.

Ce test ne dit pas non plus si le débruitage améliore un portefeuille de variance
minimale, des ratios de couverture ou l'attribution du risque. Ces applications
utilisent directement la covariance. Ici, la matrice est comprimée en quelques
scalaires avant d'entrer dans un modèle de prévision. Cette transformation change
la question de recherche.

## L'expérience suivante

Je séparerais les deux hypothèses. D'abord, les estimateurs de covariance seraient
évalués selon leurs propres objectifs : variance de portefeuille hors échantillon,
rotation des poids et sensibilité à la fenêtre d'estimation. Ensuite, la prévision
de variance suivrait une échelle de benchmarks : dernière valeur, modèle
hétérogène autorégressif de volatilité, puis modèles régularisés avec réglage
temporel imbriqué. Un univers plus large donnerait aussi davantage de sens au
cadre RMT, car le ratio $N/T$ deviendrait moins indulgent.

Le résultat actuel constitue un bon point d'arrêt. Une estimation du risque peut
être mieux conditionnée et utile économiquement sans devenir un meilleur signal
prédictif. Tester ces deux propositions séparément évite d'accorder à une belle
matrice un mérite qu'elle n'a pas encore gagné.

## Reproductibilité

Le calcul repose sur le cache de cours ajustés suivi dans le dépôt, du 2008-01-02
au 2024-12-31. Les métriques et prévisions figées de l'article se trouvent sous
`blog/data/`, et `blog/generate_charts.py` régénère les deux graphiques. La
couverture a été créée pour cet article avec un modèle de génération d'images.
L'étude est une démonstration à paramètres fixes, pas une recherche parmi
plusieurs univers, horizons, pénalités ou ensembles de features.
