# 🌾 AgriSmart Burundi — Prédiction des Récoltes par IA
## 🎯 Présentation du projet

**AgriSmart Burundi** est une application web d'intelligence artificielle qui prédit si une récolte sera **bonne ou mauvaise** à partir de données climatiques et agronomiques. Le projet couvre **15 provinces** du Burundi, **6 cultures** principales, sur la période **2015–2023** (1 620 observations).

L'application compare trois algorithmes de Machine Learning et fournit des recommandations agronomiques personnalisées basées sur les conditions saisies par l'utilisateur.

### 🚀 Démo en ligne

👉 **https://agriai-hvnkkmbmqfmtdrdt9x5wpp.streamlit.app/**



## 📸 Aperçu de l'interface

L'application est composée de deux onglets :

| Onglet | Description |
|--------|-------------|
| 🔮 **Prédiction** | Formulaire de saisie des paramètres + résultat de prédiction + comparaison des 3 modèles + recommandations |
| 📊 **Performances des Modèles** | Métriques détaillées, graphiques comparatifs Accuracy / AUC-ROC |



## 🏗️ Structure du projet


Agriculture_IA/
├── app.py                        # Application Streamlit principale
├── requirements.txt              # Dépendances Python
├── TP_Agriculture_Burundi.ipynb  # Notebook d'entraînement complet (50 cellules)
├── README.md                     # Ce fichier
└── modeles/                      # Modèles ML pré-entraînés
    ├── arbre_decision.pkl        # Arbre de Décision (max_depth=4)
    ├── foret_aleatoire.pkl       # Forêt Aléatoire (100 arbres)
    ├── regression_logistique.pkl # Régression Logistique (lbfgs)
    ├── scaler.pkl                # StandardScaler (normalisation)
    ├── feature_names.pkl         # Liste des features après encodage
    ├── num_features.pkl          # Features numériques à normaliser
    └── metrics.pkl               # Métriques de performance sauvegardées


## 📊 Données & Features

### Variables d'entrée

| Feature | Type | Description |
|---------|------|-------------|
| `province` | Catégoriel | 15 provinces du Burundi |
| `culture` | Catégoriel | Maïs, Haricot, Manioc, Patate douce, Sorgho, Bananier |
| `saison` | Catégoriel | A (Mars–Juin) ou B (Septembre–Décembre) |
| `altitude_m` | Numérique | Altitude en mètres (700–2100 m) |
| `pluviometrie_mm` | Numérique | Pluviométrie en mm (200–1300 mm) |
| `temperature_moy_C` | Numérique | Température moyenne en °C (15–28°C) |
| `superficie_ha` | Numérique | Superficie cultivée en hectares (0.3–5 ha) |
| `nb_menages` | Numérique | Nombre de ménages concernés |
| `utilisation_engrais` | Binaire | Utilisation d'engrais (0/1) |
| `acces_irrigation` | Binaire | Accès à l'irrigation (0/1) |

### Variable cible

   bonne_recolte` : **1** = Bonne récolte, **0** = Mauvaise récolte  
  ⚠️ Dataset déséquilibré : ~93% de bonnes récoltes — l'**AUC-ROC** est la métrique prioritaire.



## 🤖 Modèles de Machine Learning

Trois algorithmes sont entraînés, comparés et disponibles dans l'application :

### 1. 🌳 Arbre de Décision
- Paramètres : `max_depth=4`, `criterion='gini'`
- Avantage : interprétable, visualisable
- Limite : tendance à l'overfitting sur les arbres profonds

### 2. 🌲 Forêt Aléatoire *(recommandé)*
- Paramètres : `n_estimators=100`, `random_state=42`
- Principe : **Bagging** — combine 100 arbres entraînés sur sous-échantillons aléatoires
- Meilleure accuracy globale

### 3. 📈 Régression Logistique
- Paramètres : `max_iter=1000`, `solver='lbfgs'`
- Avantage : meilleure **AUC-ROC**, meilleure discrimination entre les deux classes

### 📈 Performances (jeu de test — 20% des données, 324 observations)

| Modèle | Accuracy | AUC-ROC |
|--------|----------|---------|
| Arbre de Décision | 89.87% | 0.7621 |
| **Forêt Aléatoire** | **93.35%** | 0.7450 |
| Régression Logistique | 93.67% | **0.8303** |

> 💡 **Interprétation** : La Régression Logistique obtient la meilleure AUC-ROC (0.83), métrique prioritaire sur un dataset déséquilibré. La Forêt Aléatoire reste le modèle par défaut pour son robustesse.



## ⚙️ Pipeline de traitement des données


Données brutes
    ↓
Imputation des valeurs manquantes (médiane par province pour pluviométrie)
    ↓
Encodage One-Hot des variables catégorielles (province, culture, saison)
    ↓
Normalisation StandardScaler (variables numériques continues)
    ↓
Split Train/Test 80/20 stratifié (random_state=42)
    ↓
Entraînement des 3 modèles
    ↓
Évaluation : Accuracy, F1-score, AUC-ROC, Matrice de confusion
    ↓
Sauvegarde .pkl (joblib)


## 🖥️ Installation et lancement local

### Prérequis
- Python 3.8+
- pip

### Étapes


# 1. Cloner le dépôt
git clone https://github.com/morialouange/Agri_IA.git
cd Agriculture_IA

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'application
streamlit run app.py

L'application s'ouvre automatiquement à l'adresse `http://localhost:8501`

### Dépendances (`requirements.txt`)


streamlit
scikit-learn
pandas
numpy
matplotlib
joblib


## 🔮 Utilisation de l'application

### Onglet Prédiction

1. **Localisation & Culture** — choisissez la province, la culture et la saison agricole
2. **Conditions climatiques** — renseignez l'altitude, la pluviométrie et la température
3. **Parcelle & Intrants** — superficie, nombre de ménages, utilisation d'engrais et d'irrigation
4. **Algorithme** — sélectionnez le modèle de prédiction (Forêt Aléatoire recommandé)
5. Cliquez sur **⟶ Lancer la prédiction**

L'application affiche :
- Le résultat principal (Bonne / Mauvaise récolte) avec la probabilité
- Une jauge de confiance du modèle
- La comparaison des 3 algorithmes sur les mêmes paramètres
- Des **recommandations agronomiques** personnalisées (irrigation, engrais, choix de culture)

### Recommandations automatiques

L'application génère des conseils contextuels selon les conditions :
- Pluviométrie < 600 mm → suggestion de collecte d'eau ou d'irrigation
- Sans engrais → impact potentiel sur le rendement signalé
- Haricot avec pluviométrie < 650 mm → alternative Sorgho ou Manioc proposée
- Température > 24°C pour Maïs/Haricot → variétés thermotolérantes conseillées

---

## 📓 Notebook d'entraînement

Le fichier `TP_Agriculture_Burundi.ipynb` contient l'intégralité de la démarche scientifique organisée en 6 exercices :

| Exercice | Contenu |
|----------|---------|
| **Ex. 1** | Chargement, exploration, qualité des données, visualisations |
| **Ex. 2** | Prétraitement, encodage, normalisation, split train/test |
| **Ex. 3** | Arbre de Décision — entraînement, matrice de confusion, importance des variables, analyse overfitting |
| **Ex. 4** | Forêt Aléatoire — bagging, validation croisée 5-folds, impact du nombre d'arbres |
| **Ex. 5** | Régression Logistique — courbes ROC, comparaison AUC des 3 modèles |
| **Ex. 6** | Prédiction sur de nouveaux scénarios agricoles, interprétation, recommandations |

---

## 🗺️ Provinces couvertes

Bujumbura Rural · Gitega · Ngozi · Muyinga · Kirundo · Kayanza · Muramvya · Mwaro · Bubanza · Cibitoke · Makamba · Rutana · Ruyigi · Cankuzo · Bururi

## 🌱 Cultures supportées

Maïs · Haricot · Manioc · Patate douce · Sorgho · Bananier



