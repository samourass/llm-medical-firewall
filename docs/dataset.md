# Dataset

## Résumé

- **100 % synthétique**, généré par des gabarits (`src/llmfw/data/templates.py`), graine fixe
  (`RANDOM_SEED=42`). Aucune donnée réelle (patient, mot de passe, clé API) n'est utilisée.
- **6 566 exemples** après déduplication et filtrage des quasi-doublons — dépasse largement le
  minimum de 2 000 exemples demandé.
- **6 classes** (colonne `category`) : `benign`, `prompt_injection`, `jailbreak`,
  `rag_prompt_injection`, `pii_exfiltration`, `secret_exfiltration`.
- Une colonne `label` binaire (0 = benign, 1 = attaque, toute catégorie confondue) coexiste avec
  `category` : elle sert aux modèles binaires historiques (`training/classical.py`, étapes 3-5).
  Les nouveaux modèles multiclasses (`training/multiclass.py`) utilisent `category` directement.

## Colonnes

| Colonne | Description |
|---|---|
| `text` | La requête utilisateur (anglais) |
| `label` | 0 = benign, 1 = attaque (toute catégorie) — utilisé par les modèles binaires historiques |
| `category` | Une des 6 classes — utilisé par les modèles multiclasses et le firewall unifié |
| `subtype` | Sous-catégorie plus fine (ex. `benign_user`, `benign_hard`, `benign_doc`) |
| `template_id` | Identifiant du gabarit ayant généré l'exemple, utilisé pour l'anti-fuite |

## Répartition par catégorie

| Catégorie | Total | Train | Validation | Test |
|---|---|---|---|---|
| benign | 2 401 | 1 706 | 334 | 361 |
| prompt_injection | 840 | 560 | 140 | 140 |
| jailbreak | 840 | 560 | 140 | 140 |
| rag_prompt_injection | 840 | 560 | 140 | 140 |
| pii_exfiltration | 840 | 560 | 140 | 140 |
| secret_exfiltration | 805 | 535 | 140 | 130 |

## Split et anti-fuite

- **Split stratifié par sous-type, disjoint par gabarit** (`template_id`) : un gabarit n'apparaît
  que dans un seul split. Cible 70/15/15 ; ratio réel 68,2 % / 15,7 % / 16,0 % (des groupes
  entiers de gabarits sont affectés, d'où le léger écart).
- Les exemples de validation/test dont la similarité TF-IDF avec le train dépasse 0,90 sont
  retirés (anti quasi-doublon).
- Audit (`data/processed/dataset_card.json`) : **0 doublon exact, 0 gabarit partagé entre
  splits**.
- Validation par `tests/test_dataset.py` : génération déterministe, labels cohérents avec
  `category`, absence de doublon après normalisation, absence de fuite train/val/test.

## Regénérer le dataset

```powershell
# Dossier : llm-medical-firewall\
.\scripts\01_build_dataset.ps1
```

## Limites connues (héritées de la version binaire, toujours valables en multiclasse)

- Chaque catégorie d'attaque n'a que 4 gabarits dans le test (~140 exemples) : un seul gabarit
  raté fait chuter le recall d'une catégorie d'environ 25 points.
- Données synthétiques : les scores sont optimistes par rapport à des attaques réelles.
- Le vocabulaire des « slots » (médicaments, chaînes canari type `PWNED`, `ACCESS GRANTED`) est
  partagé entre splits — les résultats ne présument pas d'une généralisation à un vocabulaire
  totalement inédit.
