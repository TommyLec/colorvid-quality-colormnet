# NOTICE — moteur « quality » ColorMNet

Ce module contient des portions **adaptées** de :

> **ColorMNet: A Memory-based Deep Spatial-Temporal Feature Propagation Network for
> Video Colorization** — Yixin Yang, Jiangxin Dong, Jinhui Tang, Jinshan Pan (ECCV 2024)
> <https://github.com/yyang181/colormnet>

## Licence

L'œuvre amont est diffusée sous **Creative Commons Attribution – Pas d'Utilisation
Commerciale – Partage dans les Mêmes Conditions 4.0 International (CC BY-NC-SA 4.0)** ;
le texte intégral est fourni par le dépôt amont dans son fichier `LICENSES`.

**Obligations** (résumé opérationnel, cf. le `README.md` de ce dépôt) :

1. **Attribution** — créditer les auteurs, indiquer la source et la licence, et
   **signaler les modifications** apportées.
2. **NonCommercial** — aucune exploitation à finalité commerciale tant que ce code
   ou les poids associés sont présents dans le binaire distribué.
3. **ShareAlike** — toute œuvre dérivée (y compris notre export ONNX et nos
   adaptations) reste sous CC BY-NC-SA 4.0.

Composants tiers utilisés par l'amont, dont les licences sont compatibles :
**XMem** (MIT), **DINOv2** (Apache-2.0, code et poids).

## Modifications apportées par ce dépôt

| Modification | Nature |
| :--- | :--- |
| Export ONNX des quatre sous-graphes (`encode_key`, `short_term_attn`, `segment`, `encode_value`) | adaptation |
| Remplacement de l'opérateur de corrélation `spatial_correlation_sampler` par une implémentation en primitives standard, validée à 1,5·10⁻⁵ contre l'originale | adaptation |
| Extraction de la récurrence (mémoire, calendrier de pas) hors du graphe, côté Rust | adaptation |
| Réécriture Rust de la lecture mémoire et du suivi par objet, vérifiée contre l'exécution de référence | adaptation |
| Pilote d'inférence CPU et outillage d'évaluation (`.j0/`, non livrés) | outillage |
| Outils d'évaluation **dans le dépôt** : `examples/quality_ab.rs` (A/B dans le pipeline média réel) et les tests `orchestration_matches_pytorch.rs` / `batch_reference.rs` | outillage |

Les outils « dans le dépôt » vivent **dans ce dossier** : ils disparaissent avec
lui. `examples/quality_ab.rs` ajoute `colorvid-media` aux dépendances **de
développement** uniquement — le moteur lui-même n'en dépend pas, et le build par
défaut du produit ne compile ni cette crate ni cet exemple.

## Cas de test (`tests/fixtures/`)

Les trois fichiers JSON de `tests/fixtures/` sont des **vecteurs numériques**
(tenseurs d'entrée et sorties attendues) **produits en exécutant le code amont**
(`model/memory_util.py`, `inference/memory_manager.py`,
`inference/kv_memory_store.py`) ; ils ne contiennent aucune ligne de code amont.
Ils servent d'oracle aux tests unitaires et sont, par précaution, **traités comme
des dérivés** : ils vivent dans ce module isolé, donc sous CC BY-NC-SA 4.0 comme
le reste de l'îlot, et disparaissent avec lui (cf. le dépôt cœur (`docs/plan-colormnet-quality.md`)
§9). Ils ne sont inclus que dans les **binaires de test** (`include_str!`), jamais
dans un binaire livré.

## À faire avant toute distribution (jalon C6)

- [ ] embarquer le **texte intégral** de la CC BY-NC-SA 4.0 à côté des poids ;
- [ ] porter l'attribution dans l'écran « À propos » et le sélecteur de moteur ;
- [ ] vérifier qu'aucun artefact « quality » n'est présenté comme commercial.
