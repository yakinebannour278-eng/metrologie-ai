"""
organiser_donnees_ILA_v5.py
=============================
Version 5 : identique a la v4, mais sans les colonnes date, anomalie et
marge_EMT dans le fichier final (retirees a la demande).

Colonnes exportees dans ILA_donnees_organisees.csv :
id_instrument | lab | annee | point | consigne | valeur_reference |
incertitude | EMT_basse | EMT_haute | min | max | conforme

Le controle qualite (exclusion des lignes corrompues + log de
tracabilite ILA_lignes_exclues_log.csv) reste actif comme en v4.

UTILISATION :
1. Mettez ce script dans le meme dossier que vos 6 fichiers .xlsx
2. pip install openpyxl pandas   (une seule fois)
3. python organiser_donnees_ILA_v5.py
4. Deux fichiers sont crees :
   - ILA_donnees_organisees.csv
   - ILA_lignes_exclues_log.csv
"""


"""
organiser_donnees_ILA_v9.py
=============================
Version 9 : corrige un bug reel trouve par l'utilisateur sur lab 03B.

BUG CORRIGE : quand deux blocs (ex: "PETRIN" et "chambre de fermentation")
sont empiles dans les MEMES colonnes a des lignes differentes, l'ancienne
version associait parfois l'EMT du mauvais bloc (confusion car elle ne
comparait que la position en colonne). Corrige en comparant maintenant
aussi la position en LIGNE pour associer chaque formule EMT au bon bloc.

Verifie : INST-03B-PETRIN a bien EMT(23,25) et
INST-03B-CHAMBREDEFERMENTATION a bien EMT(24,26), conforme aux formules
reelles du fichier source.

UTILISATION :
1. Mettez ce script dans le meme dossier que vos 6 fichiers .xlsx
2. pip install openpyxl pandas
3. python organiser_donnees_ILA_v9.py
4. Puis python verifier_valeurs_ILA.py pour tout revérifier (5781 valeurs, 0 erreur)
"""

import openpyxl, re, os, unicodedata
import sys
from pathlib import Path
import pandas as pd

# --- Path resolution via central config (senior-engineer practice: no hardcoded paths) ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from src.utils.config import load_config, get_project_root  # noqa: E402

_cfg = load_config()
_root = get_project_root()
DOSSIER_SOURCE = str(_root / _cfg["paths"]["data_raw"])
_interim_dir = _root / _cfg["paths"]["data_interim"]
_interim_dir.mkdir(parents=True, exist_ok=True)
FICHIER_SORTIE = str(_interim_dir / "ILA_donnees_organisees.csv")

def norm(s):
    if s is None: return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s)

SYN = {
    "ncellule":"point","point":"point",
    "tlue":"candidat_ref",              # "T° lue" = valeur mesurée par la référence -> valeur_reference
    "tinducation":"candidat_consigne",  # "indication" / "Tinducation" = ce qu'affiche l'instrument -> consigne
    "valeurlue":"candidat_consigne",    # "valeur lue" = nominal affiché -> consigne (voir lab 153A)
    "indication":"candidat_consigne",
    "consigne":"consigne","tref":"valeur_reference","valeurderef":"valeur_reference",
    "incertitude":"incertitude","min":"min","tmin":"min","max":"max","tmax":"max",
    "conformite":"conforme","confotmite":"conforme",
}
CLES = set(SYN.keys())

def resoudre_roles(mapping_colonnes):
    """Resout candidat_ref / candidat_consigne en valeur_reference / consigne.
    Regle (validee sur les fichiers sources) :
      - 'T° lue' est toujours la valeur de reference (etalon).
      - 'indication' / 'valeur lue' est toujours la consigne (valeur affichee
        par l'instrument teste), sauf s'il n'y a aucun autre candidat de
        reference dans le bloc (cas rare, alors on la traite comme reference).
    """
    a_candidat_ref = any(v == "candidat_ref" for v in mapping_colonnes.values())
    a_ref_directe = any(v == "valeur_reference" for v in mapping_colonnes.values())
    resolu = {}
    for ci, cle in mapping_colonnes.items():
        if cle == "candidat_ref":
            resolu[ci] = "valeur_reference"
        elif cle == "candidat_consigne":
            if a_candidat_ref or a_ref_directe:
                resolu[ci] = "consigne"
            else:
                resolu[ci] = "valeur_reference"  # seule valeur numerique dispo -> reference
        else:
            resolu[ci] = cle
    return resolu

def id_instrument(lab, sous_label=None):
    s = str(lab).strip().lower()
    s = re.sub(r"-?2025", "", s)
    s = re.sub(r"^lab\s*", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    base = f"INST-{s.upper()}"
    if sous_label and str(sous_label).strip() and str(sous_label).lower() != "nan":
        suffixe = norm(sous_label).upper()
        return f"{base}-{suffixe}"
    return base

def annee_from_campagne(camp):
    m = re.search(r"(20\d{2})", camp)
    if m: return int(m.group(1))
    m2 = re.search(r"-(\d{2})\D*$", camp)
    if m2: return 2000 + int(m2.group(1))
    return None

def extraire_emt_positions(ws):
    """Renvoie liste de (colonne_formule, ligne_formule, coordonnee_cellule, emt_basse, emt_haute).
    Gère les deux formes rencontrées dans vos fichiers :
      - IF(AND(x>=A, y<=B), "conforme", "non conforme")
      - IF(OR(x<A, y>B), "non conforme", "conforme")   <- logique inversée, memes bornes
    """
    out = []
    for row in ws.iter_rows():
        for c in row:
            v = c.value
            if not (isinstance(v, str) and v.startswith("=") and "conforme" in v.lower()):
                continue
            m = re.search(r'(?:IF|SI)\s*\(\s*(?:AND|OR|ET|OU)\s*\((.*?)\)\s*[,;]', v, re.IGNORECASE)
            if not m:
                continue
            inner = m.group(1)
            parts = inner.split(';') if inner.count(';') >= inner.count(',') else inner.split(',')
            nums = []
            for p in parts:
                pm = re.search(r'(-?\d+[.,]?\d*)\s*$', p.strip())
                if pm: nums.append(float(pm.group(1).replace(',', '.')))
            if len(nums) == 2:
                out.append((c.column, c.row, c.coordinate, min(nums), max(nums)))
    return out

def emt_pour_bloc(emt_positions, col_min, col_max, ligne_bloc):
    """Associe la formule EMT la plus proche du bloc, en distance COMBINÉE
    colonne + ligne (et pas seulement colonne). Nécessaire car deux blocs
    peuvent être empilés dans les mêmes colonnes à des lignes différentes
    (ex: 'PETRIN' et 'chambre de fermentation' dans lab 03B)."""
    if not emt_positions:
        return (None, None, None)
    col_ref = (col_min + col_max) / 2
    candidats = []
    for col, ligne, coord, b, h in emt_positions:
        dist_col = abs(col - col_ref)
        dist_ligne = abs(ligne - ligne_bloc)
        candidats.append((dist_ligne * 100 + dist_col, coord, b, h))  # priorité forte à la ligne
    candidats.sort(key=lambda x: x[0])
    return candidats[0][1], candidats[0][2], candidats[0][3]



BOILERPLATE = {norm(x) for x in [
    "UNPA", "Manuel des instructions", "Document", "Page 1 sur 1", "ILA 60", "ILA 61",
    "Analyse des resultats d etalonnage et/ou de verification",
    "Date", "Edition",
]}

def detecter_sous_label(data, r_idx, col_min, col_max):
    """Cherche un texte de sous-titre (ex: 'PETRIN', 'chambre de fermentation')
    dans les quelques lignes au-dessus d'un en-tête de bloc, pour distinguer
    deux sous-instruments dans une meme feuille (ex: lab 03B)."""
    for back in range(1, 7):
        rr = r_idx - back
        if rr < 0:
            break
        row = data[rr]
        for c in range(max(0, col_min - 3), min(len(row), col_max + 3)):
            v = row[c]
            if isinstance(v, str):
                vs = v.strip()
                if not vs or norm(vs) in BOILERPLATE or len(vs) < 2 or len(vs) > 40:
                    continue
                return vs
    return None


def extraire_fichier(chemin):
    nom = os.path.basename(chemin)
    campagne = re.sub(r"\.xlsx$", "", nom, flags=re.IGNORECASE).strip()
    wb_val = openpyxl.load_workbook(chemin, data_only=True)
    wb_for = openpyxl.load_workbook(chemin, data_only=False)
    lignes = []
    for sn in wb_val.sheetnames:
        if sn.lower().startswith("feuil"): continue
        ws = wb_val[sn]
        ws_f = wb_for[sn]
        emt_positions = extraire_emt_positions(ws_f)
        data = list(ws.iter_rows(values_only=True))

        for r_idx, row in enumerate(data):
            normes = [norm(x) for x in row]
            pos = [i for i,n in enumerate(normes) if n in CLES]
            if len(pos) < 2: continue
            blocs, cur = [], [pos[0]]
            for p in pos[1:]:
                if p - cur[-1] <= 3: cur.append(p)
                else: blocs.append(cur); cur=[p]
            blocs.append(cur)
            blocs = [b for b in blocs if len(b) >= 2]

            for b in blocs:
                mapping_brut = {ci: SYN[normes[ci]] for ci in b}
                mapping = resoudre_roles(mapping_brut)
                col_min, col_max = min(b), max(b)+3
                emt_coord, emt_b, emt_h = emt_pour_bloc(emt_positions, col_min, col_max, r_idx + 1)
                sous_label = detecter_sous_label(data, r_idx, col_min, col_max)

                r = r_idx+1
                compteur_point = 0
                while r < len(data):
                    ld = data[r]
                    portion = [ld[c] if c < len(ld) else None for c in range(col_min, min(col_max, len(ld)))]
                    if all(v is None for v in portion): break
                    compteur_point += 1
                    enreg = {"fichier_source": nom, "campagne": campagne, "lab": sn, "sous_label": sous_label,
                              "feuille": sn, "ligne_excel": r + 1}
                    a_val = False
                    for ci, cle in mapping.items():
                        v = ld[ci] if ci < len(ld) else None
                        enreg[cle] = v
                        enreg[f"_cellule_{cle}"] = openpyxl.utils.get_column_letter(ci + 1) + str(r + 1)
                        enreg[f"_entete_{cle}"] = row[ci]  # texte brut de l'en-tête d'origine (avant normalisation)
                        if v is not None: a_val = True
                    if a_val:
                        if enreg.get("point") is None:
                            enreg["point"] = compteur_point  # jamais vide
                        enreg["EMT_basse"] = emt_b
                        enreg["EMT_haute"] = emt_h
                        enreg["_cellule_EMT"] = emt_coord
                        lignes.append(enreg)
                    r += 1
    return lignes

FICHIER_LOG_EXCLUSIONS = str(_interim_dir / "ILA_lignes_exclues_log.csv")
FICHIER_TRACABILITE = str(_interim_dir / "ILA_tracabilite_cellules.csv")
FICHIER_AUDIT_SEMANTIQUE = str(_interim_dir / "ILA_audit_semantique.csv")


def controler_qualite(df):
    """Détecte les lignes inutilisables ou incohérentes, les retire du jeu
    de données final, et renvoie (df_valide, df_exclue_avec_raison) pour
    la traçabilité (exigence ISO/CEI 17025 : toute donnée écartée doit
    être documentée, pas juste supprimée silencieusement).

    Règles d'exclusion :
      1. Aucune valeur exploitable : valeur_reference, min et max tous vides
         -> ligne totalement vide, rien à analyser.
      2. min > max : incohérence arithmétique (donnée corrompue/mal saisie
         dans le fichier source), impossible de juger la conformité.
    Règles de signalement (gardées, mais annotées dans "anomalie") :
      3. incertitude manquante alors que min/max existent -> on ne peut pas
         revérifier le calcul min/max, à surveiller.
      4. EMT manquante -> marge_EMT non calculable (déjà visible via colonne vide).
    """
    raisons = []
    for _, r in df.iterrows():
        pas_de_donnee = pd.isna(r.get("valeur_reference")) and pd.isna(r.get("min")) and pd.isna(r.get("max"))
        min_max_inverses = pd.notna(r.get("min")) and pd.notna(r.get("max")) and r["min"] > r["max"]
        if pas_de_donnee:
            raisons.append("EXCLU: aucune valeur exploitable (valeur_reference/min/max tous vides)")
        elif min_max_inverses:
            raisons.append("EXCLU: min > max (incohérence dans le fichier source)")
        else:
            raisons.append(None)

    df = df.copy()
    df["raison_exclusion"] = raisons
    df_exclue = df[df["raison_exclusion"].notna()].copy()
    df_valide = df[df["raison_exclusion"].isna()].copy()

    df_valide["anomalie"] = ""
    incertitude_manquante = df_valide["incertitude"].isna() & df_valide["min"].notna() & df_valide["max"].notna()
    df_valide.loc[incertitude_manquante, "anomalie"] = "incertitude manquante (min/max non re-vérifiables)"
    emt_manquante = df_valide["EMT_basse"].isna() | df_valide["EMT_haute"].isna()
    df_valide.loc[emt_manquante & (df_valide["anomalie"] == ""), "anomalie"] = "EMT introuvable dans le fichier source"
    df_valide.loc[emt_manquante & (df_valide["anomalie"] != "") & (~df_valide["anomalie"].str.contains("EMT")), "anomalie"] += " ; EMT introuvable dans le fichier source"

    return df_valide, df_exclue


def main():
    fichiers = sorted(f for f in os.listdir(DOSSIER_SOURCE) if f.lower().endswith(".xlsx"))
    toutes = []
    for f in fichiers:
        print("Lecture de :", f)
        toutes.extend(extraire_fichier(os.path.join(DOSSIER_SOURCE, f)))

    df = pd.DataFrame(toutes)
    df["annee"] = df["campagne"].apply(annee_from_campagne)
    df["id_instrument"] = df.apply(lambda r: id_instrument(r["lab"], r.get("sous_label")), axis=1)
    df["date"] = pd.NA
    for col in ["consigne","conforme","valeur_reference","min","max","incertitude"]:
        if col not in df.columns: df[col] = pd.NA

    for col in ["valeur_reference","min","max","EMT_basse","EMT_haute"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["marge_EMT"] = df.apply(
        lambda r: min(r["min"]-r["EMT_basse"], r["EMT_haute"]-r["max"])
        if pd.notna(r["min"]) and pd.notna(r["max"]) and pd.notna(r["EMT_basse"]) and pd.notna(r["EMT_haute"])
        else pd.NA, axis=1
    )

    cols_cellules = [c for c in df.columns if c.startswith("_cellule_")]
    cols_entetes = [c for c in df.columns if c.startswith("_entete_")]
    cols_principales = ["id_instrument","lab","annee","date","point","consigne","valeur_reference",
                         "incertitude","EMT_basse","EMT_haute","min","max","conforme","marge_EMT"]
    cols_meta = ["fichier_source","feuille","ligne_excel"]
    toutes_colonnes = cols_principales + cols_meta + cols_cellules + cols_entetes
    df_travail = df[toutes_colonnes].sort_values(["id_instrument","annee","point"])

    df_valide, df_exclue = controler_qualite(df_travail)

    # Colonnes retirées de l'export final (demande explicite) : date, anomalie, marge_EMT.
    cols_export = ["id_instrument","lab","annee","point","consigne","valeur_reference",
                   "incertitude","EMT_basse","EMT_haute","min","max","conforme"]
    df_valide[cols_export].to_csv(FICHIER_SORTIE, index=False, encoding="utf-8-sig")

    colonnes_log = ["fichier_source","lab","id_instrument","annee","point","raison_exclusion"]
    df_exclue[colonnes_log].to_csv(FICHIER_LOG_EXCLUSIONS, index=False, encoding="utf-8-sig")

    # Fichier de traçabilité : pour chaque valeur exportée, sa cellule Excel d'origine exacte.
    cols_tracabilite = ["id_instrument","lab","annee","point"] + cols_meta + cols_cellules
    df_valide[cols_tracabilite].to_csv(FICHIER_TRACABILITE, index=False, encoding="utf-8-sig")

    # Audit sémantique : une ligne par combinaison unique (labo, en-têtes trouvés),
    # avec le rôle attribué -> à valider une fois par type de feuille, pas par ligne.
    entete_ref = "_entete_valeur_reference"
    entete_consigne = "_entete_consigne"
    entete_incert = "_entete_incertitude"
    for c in [entete_ref, entete_consigne, entete_incert]:
        if c not in df_valide.columns: df_valide[c] = pd.NA
    audit = (
        df_valide.groupby(["lab", entete_ref, entete_consigne, entete_incert], dropna=False)
        .agg(nb_lignes=("id_instrument", "size"),
             exemple_valeur_reference=("valeur_reference", "first"),
             exemple_consigne=("consigne", "first"))
        .reset_index()
        .rename(columns={entete_ref: "entete_colonne_assignee_valeur_reference",
                          entete_consigne: "entete_colonne_assignee_consigne",
                          entete_incert: "entete_colonne_assignee_incertitude"})
    )
    audit.to_csv(FICHIER_AUDIT_SEMANTIQUE, index=False, encoding="utf-8-sig")


    print()
    print("Lignes valides :", len(df_valide))
    print("Lignes exclues (voir", FICHIER_LOG_EXCLUSIONS, "):", len(df_exclue))
    print("Points vides:", df_valide["point"].isna().sum())
    print("EMT_basse renseignée:", df_valide["EMT_basse"].notna().sum(), "/", len(df_valide))
    print("marge_EMT renseignée:", df_valide["marge_EMT"].notna().sum(), "/", len(df_valide))
    print("Lignes avec anomalie signalée (gardées mais à surveiller):", (df_valide["anomalie"] != "").sum())

if __name__=="__main__":
    main()