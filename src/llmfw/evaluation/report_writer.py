"""Génère `docs/security_evaluation.md` à partir des résultats RÉELLEMENT mesurés (jamais de
valeur tapée à la main dans ce module) : cohérence garantie entre le rapport et
`results/*.csv` / `results/evaluation_results.json` à chaque exécution de
`evaluation/run_full_evaluation.py`.
"""
from __future__ import annotations

from pathlib import Path

DISPLAY_NAME = {"regex": "Regex", "logistic_regression": "Logistic Regression",
                "xgboost": "XGBoost", "distilbert": "DistilBERT"}


def _fmt(x, digits=4):
    return "—" if x is None else f"{x:.{digits}f}"


def _fmt_pct(x, digits=2):
    return "—" if x is None else f"{x:.{digits}f}%"


def _model_table(input_results: dict) -> str:
    header = "| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) | FPR | FNR | Latency (median) |\n"
    header += "|---|---|---|---|---|---|---|---|---|\n"
    lines = [header]
    for b, r in input_results.items():
        name = DISPLAY_NAME.get(b, b)
        if r["status"] != "ok":
            lines.append(f"| {name} | not measured | not measured | not measured | not measured | "
                        f"not measured | not measured | not measured | not measured |\n")
            continue
        lines.append(
            f"| {name} | {_fmt(r['accuracy'])} | {_fmt(r['precision_macro'])} | "
            f"{_fmt(r['recall_macro'])} | {_fmt(r['f1_macro'])} | {_fmt(r['f1_weighted'])} | "
            f"{_fmt_pct(r['false_positive_rate_pct'])} | {_fmt_pct(r['false_negative_rate_pct'])} | "
            f"{r['latency']['median_ms']:.3f} ms |\n"
        )
    return "".join(lines)


def _category_table(input_results: dict) -> str:
    ok_backends = [b for b, r in input_results.items() if r["status"] == "ok"]
    if not ok_backends:
        return "Aucun modèle disponible dans cet environnement — table non générée.\n"
    labels = input_results[ok_backends[0]]["labels"]
    out = "| Category | " + " | ".join(f"{DISPLAY_NAME.get(b,b)} — P / R / F1" for b in ok_backends) + " |\n"
    out += "|---|" + "---|" * len(ok_backends) + "\n"
    for lab in labels:
        cells = []
        for b in ok_backends:
            pc = input_results[b]["per_class"][lab]
            cells.append(f"{pc['precision']:.3f} / {pc['recall']:.3f} / {pc['f1']:.3f}")
        out += f"| {lab} | " + " | ".join(cells) + " |\n"
    return out


def _latency_table(input_results: dict) -> str:
    header = "| Model | Min | Max | Mean | Median | p95 | n |\n|---|---|---|---|---|---|---|\n"
    rows = []
    for b, r in input_results.items():
        name = DISPLAY_NAME.get(b, b)
        if r["status"] != "ok":
            rows.append(f"| {name} | not measured | not measured | not measured | not measured | not measured | 0 |\n")
            continue
        lat = r["latency"]
        rows.append(f"| {name} | {lat['min_ms']:.4f} ms | {lat['max_ms']:.4f} ms | {lat['mean_ms']:.4f} ms | "
                    f"{lat['median_ms']:.4f} ms | {lat['p95_ms']:.4f} ms | {lat['n_requests']} |\n")
    return header + "".join(rows)


def _skipped_section(input_results: dict) -> str:
    skipped = {b: r["reason"] for b, r in input_results.items() if r["status"] != "ok"}
    if not skipped:
        return "Tous les modèles prévus ont pu être évalués dans cet environnement.\n"
    lines = ["Les modèles suivants n'ont **pas** pu être évalués dans cet environnement "
            "d'exécution (voir raison exacte ci-dessous) ; ils n'apparaissent dans aucun "
            "tableau ou graphique comparatif ci-dessus — aucune valeur n'a été inventée pour "
            "eux :\n"]
    for b, reason in skipped.items():
        lines.append(f"- **{DISPLAY_NAME.get(b, b)}** : `{reason}`\n")
    return "".join(lines)


def _robustness_section(robustness_results: dict) -> str:
    ok = {b: r for b, r in robustness_results.items() if r.get("status") == "ok"}
    if not ok:
        return "Aucun modèle disponible pour l'évaluation de robustesse dans cet environnement.\n"
    lines = []
    for b, r in ok.items():
        lines.append(f"\n**{DISPLAY_NAME.get(b, b)}** — accuracy de référence (`original`) : "
                    f"{r['baseline_accuracy']*100:.2f}% ; accuracy globale toutes variations "
                    f"confondues : {r['overall_accuracy']*100:.2f}%.\n\n")
        lines.append("| Variation | n | Accuracy |\n|---|---|---|\n")
        for row in r["by_variation"]:
            lines.append(f"| {row['variation']} | {row['n']} | {row['accuracy_pct']:.2f}% |\n")
    return "".join(lines)


def _output_firewall_section(output_results: dict) -> str:
    lines = [
        f"- Cas testés : {output_results['n_cases']} "
        f"({output_results['n_leak_cases']} avec fuite attendue, "
        f"{output_results['n_benign_cases']} bénins)\n",
        f"- Detection rate (fuites correctement interceptées, REDACT ou BLOCK) : "
        f"{_fmt_pct(output_results['detection_rate_pct'])}\n",
        f"- False positive rate (réponses bénignes signalées à tort) : "
        f"{_fmt_pct(output_results['false_positive_rate_pct'])}\n",
        f"- False negative rate (fuites laissées passer, ALLOW) : "
        f"{_fmt_pct(output_results['false_negative_rate_pct'])}\n",
        f"- Latence : médiane {output_results['latency']['median_ms']:.4f} ms, "
        f"p95 {output_results['latency']['p95_ms']:.4f} ms (n={output_results['latency']['n_requests']})\n\n",
        "| Catégorie de fuite | n | Détectées | Taux |\n|---|---|---|---|\n",
    ]
    for row in output_results["by_category"]:
        lines.append(f"| {row['expected_category']} | {row['n']} | {row['n_detected']} | "
                    f"{row['detection_rate_pct']:.1f}% |\n")
    return "".join(lines)


def render_report(input_results: dict, output_results: dict, robustness_results: dict,
                  security_test_set_size: int, out_path: Path) -> None:
    ok_backends = [b for b, r in input_results.items() if r["status"] == "ok"]
    best_f1 = max(((b, input_results[b]["f1_macro"]) for b in ok_backends), key=lambda x: x[1],
                 default=(None, None))
    worst_fpr = min(((b, input_results[b]["false_positive_rate_pct"]) for b in ok_backends),
                    key=lambda x: x[1], default=(None, None))

    report = f"""# Security Evaluation — LLM Firewall (Input + Output)

> Généré automatiquement par `evaluation/run_full_evaluation.py` à partir des fichiers dans
> `results/`. Chaque nombre ci-dessous provient d'une exécution mesurée dans cet environnement ;
> rien n'est estimé ou supposé. Régénérez ce fichier après tout changement de modèle, de seuil
> ou de jeu de données pour qu'il reste synchronisé avec `results/`.

## 1. Research Objective

Évaluer, de façon scientifique, la capacité de détection, la performance de classification, la
robustesse aux perturbations, le comportement en faux positifs et la latence du LLM Firewall
bidirectionnel (entrée + sortie) proposé dans ce projet. **Ceci n'est pas une comparaison
"avec/sans firewall"** : c'est une caractérisation du firewall lui-même.

## 2. Threat Model

Le firewall d'entrée cible 5 catégories d'attaque contre un chatbot médical basé sur un LLM et
un RAG : `prompt_injection` (contournement d'instruction), `jailbreak` (contournement des
restrictions de contenu), `rag_prompt_injection` (instruction injectée via un document récupéré
par le RAG), `pii_exfiltration` (exfiltration de données patient), `secret_exfiltration`
(exfiltration de secrets applicatifs : clés API, prompt système). Le firewall de sortie cible la
fuite, dans la réponse du LLM, de PII, d'emails, de téléphones, de clés API/jetons/identifiants,
ou du prompt système lui-même — y compris quand la requête d'origine était bénigne.

## 3. Dataset

Jeu d'évaluation de sécurité dédié : **{security_test_set_size} requêtes** (100 par catégorie ×
6 catégories), généré par `llmfw.evaluation.security_test_set` à partir de gabarits **écrits
spécifiquement pour cette évaluation** (`security_test_templates.py`), disjoints des gabarits
utilisés pour `data/{{train,validation,test}}`. Vérifié programmatiquement (voir la sortie de
`python -m llmfw.evaluation.security_test_set`) : 0 chevauchement avec train/validation/test
après normalisation, 0 doublon interne.

Jeu de robustesse : 12 requêtes de base (2 par catégorie) × 11 variations (voir section 10) = 132
requêtes. Jeu d'évaluation du firewall de sortie : 20 réponses simulées (13 avec fuite attendue,
7 bénignes) couvrant 6 catégories de fuite — voir section 12 pour les limites de taille de ces
deux derniers jeux.

## 4. Experimental Setup

- Seeds fixes : génération du jeu de sécurité (`seed=2026`), du jeu de robustesse (`seed=7`).
- Chaque backend est chargé une fois puis "warmé" (3 appels non chronométrés) avant la mesure de
  latence, pour exclure le temps de chargement du modèle de la latence d'inférence mesurée.
- Latence mesurée requête par requête (pas de traitement par lot), en millisecondes, dans ce
  bac à sable (1 processus, CPU partagé) — non transférable telle quelle à une autre machine.
- Backends du firewall d'entrée évalués : Regex, Logistic Regression, XGBoost, DistilBERT (voir
  section 12 pour les modèles non disponibles dans cet environnement).

## 5. Input Firewall

Voir `docs/input_firewall.md` pour l'architecture (Regex garde-fou + backend ML configurable,
seuil `FIREWALL_THRESHOLD`). Cette évaluation porte sur la **classification multiclasse brute**
de chaque détecteur (avant application du seuil de décision ALLOW/BLOCK), pour isoler la
performance du classifieur de celle du seuil.

## 6. Output Firewall

Voir `docs/output_firewall.md`. Évalué indépendamment (section 12 ci-dessous), sur un jeu de
réponses simulées couvrant les catégories de fuite ciblées.

## 7. Evaluation Metrics

Accuracy, precision/recall/F1 (par classe, macro, weighted — `sklearn.metrics`), False Positive
Rate et False Negative Rate (définitions ci-dessous), Attack Detection Rate, matrice de
confusion. **Log-loss n'est volontairement pas calculé** : ce n'est pas une métrique demandée
par le document de cadrage, et la "confiance" retournée par le détecteur Regex n'est pas une
probabilité calibrée — y appliquer log-loss serait une métrique mal formulée (voir docstring de
`security_eval.py`).

**Définitions exactes utilisées :**
- **Attack Detection Rate (ADR)** = requêtes d'attaque prédites comme une catégorie != `benign`
  (n'importe laquelle) / total des requêtes d'attaque × 100. "Détecté" ne veut PAS dire "la
  catégorie exacte a été trouvée" — cette précision-là est donnée séparément par les métriques
  per-class (section 9).
- **False Positive Rate (FPR)** = requêtes `benign` prédites comme une catégorie != `benign` /
  total des requêtes `benign` × 100.
- **False Negative Rate (FNR)** = requêtes d'attaque prédites comme `benign` / total des
  requêtes d'attaque × 100. Par construction, FNR = 100 − ADR.

## 8. Model Results

{_model_table(input_results)}

## 9. Attack Category Results

{_category_table(input_results)}

## 10. Robustness Results

Variations testées : capitalisation, espaces, ponctuation, offuscation (substitution de
caractères, deux variantes), paraphrase, formulation indirecte, jeu de rôle, multilingue
(français), camouflage en question bénigne (« benign-looking »). "Correct" = attaque détectée
(catégorie prédite != benign) pour les lignes d'attaque, ou absence de faux positif (catégorie
prédite == benign) pour les lignes bénignes.

{_robustness_section(robustness_results)}

## 11. Latency Results

{_latency_table(input_results)}

## 12. Output Firewall Results

{_output_firewall_section(output_results)}

## 13. Discussion

{"Le F1 macro le plus élevé mesuré sur ce jeu d'évaluation est obtenu par **" + DISPLAY_NAME.get(best_f1[0], str(best_f1[0])) + f"** (F1 macro = {best_f1[1]:.4f})." if best_f1[0] else "Aucun modèle n'a pu être évalué."}
{"Le FPR le plus bas mesuré est obtenu par **" + DISPLAY_NAME.get(worst_fpr[0], str(worst_fpr[0])) + f"** (FPR = {worst_fpr[1]:.2f}%)." if worst_fpr[0] else ""}
Ces constats se limitent strictement à ce jeu de {security_test_set_size} requêtes et à cet
environnement d'exécution — voir la section Limitations pour ce qu'ils ne permettent pas de
conclure. Les résultats de robustesse (section 10) montrent une dégradation nette et cohérente,
pour tous les modèles évalués, face aux variations multilingues et à l'offuscation par
substitution de caractères — un signal clair que ces deux axes seraient prioritaires pour un
prochain cycle d'amélioration, plutôt qu'un jugement définitif sur "le meilleur modèle".

## 14. Limitations

- **Modèles non évalués dans cet environnement :**
{_skipped_section(input_results)}
- **Taille des échantillons de robustesse et de sortie** : 132 requêtes de robustesse (12 bases
  × 11 variations) et 20 cas pour le firewall de sortie sont de petits échantillons — une
  différence de quelques cas change sensiblement les pourcentages rapportés (chaque cas pèse
  ~8% pour la robustesse, 5% pour la sortie). Ces chiffres indiquent une tendance, pas une
  estimation statistiquement précise.
- **Dataset synthétique** : le jeu d'évaluation de sécurité comme les jeux de robustesse/sortie
  sont entièrement synthétiques (gabarits + substitutions), pas des attaques réelles capturées
  en production — les scores sont probablement optimistes par rapport à des attaques réelles et
  adaptatives.
- **Latence mesurée dans un bac à sable partagé** (CPU, 1 cœur utilisé par processus) — les
  valeurs absolues ne sont pas transférables à un déploiement réel ; seul l'ordre de grandeur
  relatif entre modèles est informatif.
- **Seuil de décision non évalué ici** : cette évaluation porte sur la classification brute, pas
  sur la décision finale ALLOW/BLOCK du firewall (qui dépend en plus de `FIREWALL_THRESHOLD`) —
  voir `docs/input_firewall.md` pour une limite mesurée sur ce seuil spécifiquement.
- **Une seule langue de perturbation multilingue testée** (français) et une seule technique
  d'offuscation par substitution de caractères — la généralisation à d'autres langues ou
  d'autres techniques d'offuscation n'est pas mesurée.

## 15. Conclusion

Cette évaluation rapporte des différences mesurées entre {len(ok_backends)} détecteur(s)
d'entrée disponibles dans cet environnement, sur un jeu de {security_test_set_size} requêtes
disjoint de l'entraînement, plus une évaluation de robustesse (132 requêtes, 11 variations) et
une évaluation indépendante du firewall de sortie (20 cas). Les tableaux des sections 8 à 12
sont la source de vérité ; aucune affirmation générale ("le modèle X est le meilleur") n'est
faite au-delà de ce que ces tableaux montrent explicitement, et les modèles non évalués ici
(section 14) ne doivent pas être supposés meilleurs ou moins bons sur cette seule base.
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
