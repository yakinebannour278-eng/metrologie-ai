"""
augmenter_donnees_ILA.py
==========================

BUT :
Générer des données d'entraînement supplémentaires de façon
scientifiquement correcte, à partir de vos 767 mesures réelles.

METHODE UTILISEE (pourquoi elle est correcte pour des données de métrologie,
contrairement à un GAN ou SMOTE générique qui inventeraient des valeurs
sans base physique) :

1. Séparation train/test AVANT augmentation (règle d'or, sinon fuite de
   données : le modèle "trouverait" au test des points quasi identiques
   à ceux vus à l'entraînement, et vos résultats seraient faussement bons).

2. Seul le train est augmenté. Le test reste 100% réel.

3. Le bruit ajouté n'est pas arbitraire : il est tiré d'une loi Normale
   dont l'écart-type est l'incertitude RÉELLE de la mesure d'origine
   (colonne "incertitude" de vos fichiers ILA). C'est la définition même
   de l'incertitude métrologique : la plage dans laquelle la vraie valeur
   pourrait plausiblement se trouver. Le bruit est donc physiquement
   justifié, pas inventé.

4. Les colonnes dérivées (min, max, conforme) ne sont JAMAIS copiées
   telles quelles : elles sont RECALCULÉES à partir de la valeur augmentée,
   avec les mêmes formules que vos fichiers Excel sources
   (min = valeur_reference - incertitude, max = valeur_reference + incertitude,
   conforme = EMT_basse <= min et max <= EMT_haute).

5. Les colonnes qui sont des constantes physiques de l'instrument/campagne
   (EMT_basse, EMT_haute, consigne, id_instrument, annee) ne sont jamais
   modifiées - seule valeur_reference est bruitée.

6. Chaque ligne augmentée est marquée origine="augmentee" (vs "reelle"),
   pour que vous puissiez toujours évaluer votre modèle final sur des
   données 100% réelles uniquement.

UTILISATION :
1. Mettez ce script dans le même dossier que ILA_donnees_organisees.csv
2. pip install pandas scikit-learn numpy
3. python augmenter_donnees_ILA.py
4. Produit :
   - ILA_train_reel.csv        (train, données réelles seulement)
   - ILA_train_augmente.csv    (train réel + augmenté, à utiliser pour entraîner)
   - ILA_test_reel.csv         (test, données réelles seulement - NE JAMAIS augmenter)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Path resolution via central config (senior-engineer practice: no hardcoded paths) ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from src.utils.config import load_config, get_project_root  # noqa: E402

_cfg = load_config()
_root = get_project_root()
_interim_dir = _root / _cfg["paths"]["data_interim"]
_processed_dir = _root / _cfg["paths"]["data_processed"]
_processed_dir.mkdir(parents=True, exist_ok=True)

FICHIER_ENTREE = str(_interim_dir / "ILA_donnees_organisees.csv")
FICHIER_TRAIN_REEL = str(_processed_dir / "ILA_train_reel.csv")
FICHIER_TRAIN_AUGMENTE = str(_processed_dir / "ILA_train_augmente.csv")
FICHIER_TEST_REEL = str(_processed_dir / "ILA_test_reel.csv")
FACTEUR_AUGMENTATION = 3   # nombre de variantes synthétiques générées par ligne réelle du train
GRAINE_ALEATOIRE = 42      # pour reproductibilité


def recalculer_conforme(valeur_ref, incertitude, emt_basse, emt_haute):
    if pd.isna(valeur_ref) or pd.isna(incertitude) or pd.isna(emt_basse) or pd.isna(emt_haute):
        return pd.NA
    mn = valeur_ref - incertitude
    mx = valeur_ref + incertitude
    return "conforme" if (mn >= emt_basse and mx <= emt_haute) else "non conforme"


def augmenter_une_ligne(ligne, rng):
    """Génère UNE variante synthétique plausible d'une mesure réelle,
    en bruitant valeur_reference selon sa propre incertitude, puis en
    recalculant min/max/conforme en conséquence."""
    nouvelle = ligne.copy()

    if pd.isna(ligne["valeur_reference"]) or pd.isna(ligne["incertitude"]):
        return None  # rien à augmenter si la base manque

    # bruit gaussien calibré sur l'incertitude réelle de CETTE mesure
    valeur_augmentee = rng.normal(loc=ligne["valeur_reference"], scale=ligne["incertitude"])

    nouvelle["valeur_reference"] = round(valeur_augmentee, 3)
    nouvelle["min"] = round(valeur_augmentee - ligne["incertitude"], 3)
    nouvelle["max"] = round(valeur_augmentee + ligne["incertitude"], 3)
    nouvelle["conforme"] = recalculer_conforme(
        valeur_augmentee, ligne["incertitude"], ligne["EMT_basse"], ligne["EMT_haute"]
    )
    nouvelle["origine"] = "augmentee"
    return nouvelle


def main():
    df = pd.read_csv(FICHIER_ENTREE, encoding="utf-8-sig")
    df["origine"] = "reelle"

    # ------------------------------------------------------------------
    # ETAPE 1 : separation train/test AVANT toute augmentation.
    # stratify sur "conforme" pour garder la meme proportion conforme/
    # non-conforme dans les deux ensembles.
    # ------------------------------------------------------------------
    df_utilisable = df.dropna(subset=["conforme"])
    train_reel, test_reel = train_test_split(
        df_utilisable, test_size=0.2, random_state=GRAINE_ALEATOIRE,
        stratify=df_utilisable["conforme"]
    )

    # ------------------------------------------------------------------
    # ETAPE 2 : augmentation - UNIQUEMENT sur le train
    # ------------------------------------------------------------------
    rng = np.random.default_rng(GRAINE_ALEATOIRE)
    lignes_augmentees = []
    for _, ligne in train_reel.iterrows():
        for _ in range(FACTEUR_AUGMENTATION):
            nouvelle = augmenter_une_ligne(ligne, rng)
            if nouvelle is not None:
                lignes_augmentees.append(nouvelle)

    df_augmente = pd.DataFrame(lignes_augmentees)
    train_final = pd.concat([train_reel, df_augmente], ignore_index=True)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    train_reel.to_csv(FICHIER_TRAIN_REEL, index=False, encoding="utf-8-sig")
    train_final.to_csv(FICHIER_TRAIN_AUGMENTE, index=False, encoding="utf-8-sig")
    test_reel.to_csv(FICHIER_TEST_REEL, index=False, encoding="utf-8-sig")

    print("Train réel (avant augmentation)  :", len(train_reel), "lignes")
    print("Train augmenté (réel + synthétique):", len(train_final), "lignes")
    print("Test réel (jamais augmenté)       :", len(test_reel), "lignes")
    print()
    print("Répartition conforme/non conforme :")
    print("  Train réel      :", train_reel["conforme"].value_counts().to_dict())
    print("  Train augmenté  :", train_final["conforme"].value_counts().to_dict())
    print("  Test réel       :", test_reel["conforme"].value_counts().to_dict())
    print()
    print("A RETENIR : entraînez votre modèle sur ILA_train_augmente.csv,")
    print("mais évaluez-le TOUJOURS sur ILA_test_reel.csv (données 100% réelles).")


if __name__ == "__main__":
    main()