"""
verifier_valeurs_ILA.py
=========================

BUT :
Vérifier VALEUR PAR VALEUR (pas juste le nombre de lignes) que chaque
donnée du CSV final correspond exactement à la cellule Excel d'où elle a
été extraite. Utilise le fichier ILA_tracabilite_cellules.csv (produit par
organiser_donnees_ILA_v8.py) qui indique la coordonnée exacte (ex: "D8")
de chaque valeur.

Vérifie aussi EMT_basse / EMT_haute : comme elles ne sont pas dans une
cellule simple mais cachées dans une formule (ex: =IF(AND(D8>130,E8<133))),
le script relit cette formule et la re-parse indépendamment, pour confirmer
que les nombres extraits (130 et 133) sont bien les bons.

UTILISATION :
1. Mettez ce script dans le même dossier que vos 6 .xlsx et les fichiers
   ILA_donnees_organisees.csv + ILA_tracabilite_cellules.csv
2. python verifier_valeurs_ILA.py
3. À la fin : "0 erreur" = toutes les valeurs (y compris EMT) sont vérifiées correctes.
"""

import os
import re
import openpyxl
import pandas as pd

FICHIER_CSV = "ILA_donnees_organisees.csv"
FICHIER_TRACABILITE = "ILA_tracabilite_cellules.csv"
FICHIER_RAPPORT = "ILA_rapport_verification.csv"

CHAMPS_A_VERIFIER = ["consigne", "valeur_reference", "incertitude", "min", "max", "conforme"]


def reparser_emt_depuis_formule(formule):
    """Reproduit indépendamment la logique d'extraction EMT du script principal,
    pour re-vérifier que EMT_basse/EMT_haute dans le CSV sont bien ceux de la formule."""
    if not (isinstance(formule, str) and formule.startswith("=") and "conforme" in formule.lower()):
        return None, None
    m = re.search(r'(?:IF|SI)\s*\(\s*(?:AND|OR|ET|OU)\s*\((.*?)\)\s*[,;]', formule, re.IGNORECASE)
    if not m:
        return None, None
    inner = m.group(1)
    parts = inner.split(';') if inner.count(';') >= inner.count(',') else inner.split(',')
    nums = []
    for p in parts:
        pm = re.search(r'(-?\d+[.,]?\d*)\s*$', p.strip())
        if pm: nums.append(float(pm.group(1).replace(',', '.')))
    if len(nums) == 2:
        return min(nums), max(nums)
    return None, None


def valeurs_egales(v_attendue, v_trouvee):
    """Compare deux valeurs en tolérant les petits écarts d'arrondi flottant
    et les différences de type (texte 'conforme' vs 'conforme ')."""
    if pd.isna(v_attendue) and (v_trouvee is None or v_trouvee == ""):
        return True
    if pd.isna(v_attendue) or v_trouvee is None:
        return pd.isna(v_attendue) and v_trouvee is None
    if isinstance(v_attendue, (int, float)) and isinstance(v_trouvee, (int, float)):
        return abs(float(v_attendue) - float(v_trouvee)) < 1e-6
    return str(v_attendue).strip().lower() == str(v_trouvee).strip().lower()


def main():
    if not (os.path.exists(FICHIER_CSV) and os.path.exists(FICHIER_TRACABILITE)):
        print("Fichiers requis introuvables. Lancez d'abord organiser_donnees_ILA_v8.py")
        return

    df_csv = pd.read_csv(FICHIER_CSV, encoding="utf-8-sig")
    df_trace = pd.read_csv(FICHIER_TRACABILITE, encoding="utf-8-sig")

    if len(df_csv) != len(df_trace):
        print(f"ATTENTION : {len(df_csv)} lignes dans le CSV mais {len(df_trace)} dans la traçabilité.")
        print("Les deux fichiers doivent provenir du même lancement du script. Relancez organiser_donnees_ILA_v8.py.")
        return

    fusion = pd.concat([df_csv.reset_index(drop=True), df_trace.reset_index(drop=True)], axis=1)

    classeurs_valeurs = {}   # data_only=True, pour les cellules normales
    classeurs_formules = {}  # data_only=False, pour relire les formules EMT
    erreurs = []
    n_verifiees = 0

    for _, r in fusion.iterrows():
        fichier = r["fichier_source"]
        feuille = r["feuille"]

        if fichier not in classeurs_valeurs:
            classeurs_valeurs[fichier] = openpyxl.load_workbook(fichier, data_only=True)
            classeurs_formules[fichier] = openpyxl.load_workbook(fichier, data_only=False)
        ws_val = classeurs_valeurs[fichier][feuille]
        ws_for = classeurs_formules[fichier][feuille]

        # --- champs "simples" : relire la valeur de la cellule ---
        for champ in CHAMPS_A_VERIFIER:
            cellule = r.get(f"_cellule_{champ}")
            if pd.isna(cellule):
                continue
            v_trouvee = ws_val[cellule].value
            v_attendue = r[champ]
            n_verifiees += 1
            if not valeurs_egales(v_attendue, v_trouvee):
                erreurs.append({
                    "fichier": fichier, "feuille": feuille, "cellule": cellule,
                    "champ": champ, "valeur_attendue_dans_csv": v_attendue,
                    "valeur_trouvee_dans_excel": v_trouvee,
                    "id_instrument": r["id_instrument"], "annee": r["annee"], "point": r["point"],
                })

        # --- EMT : relire la formule et la re-parser indépendamment ---
        cellule_emt = r.get("_cellule_EMT")
        if pd.notna(cellule_emt):
            formule = ws_for[cellule_emt].value
            emt_b_recalcule, emt_h_recalcule = reparser_emt_depuis_formule(formule)
            for champ, v_recalcule in [("EMT_basse", emt_b_recalcule), ("EMT_haute", emt_h_recalcule)]:
                v_attendue = r[champ]
                n_verifiees += 1
                if not valeurs_egales(v_attendue, v_recalcule):
                    erreurs.append({
                        "fichier": fichier, "feuille": feuille, "cellule": cellule_emt,
                        "champ": champ, "valeur_attendue_dans_csv": v_attendue,
                        "valeur_trouvee_dans_excel": v_recalcule,
                        "id_instrument": r["id_instrument"], "annee": r["annee"], "point": r["point"],
                    })

    print("Valeurs individuelles vérifiées :", n_verifiees, "(y compris EMT_basse/EMT_haute)")
    print("Erreurs trouvées :", len(erreurs))

    if erreurs:
        rapport = pd.DataFrame(erreurs)
        rapport.to_csv(FICHIER_RAPPORT, index=False, encoding="utf-8-sig")
        print()
        print("Détail des écarts (voir aussi", FICHIER_RAPPORT, ") :")
        print(rapport.to_string(index=False))
    else:
        print()
        print("Toutes les valeurs vérifiées (y compris EMT) correspondent exactement à leur source Excel. ✓")


if __name__ == "__main__":
    main()