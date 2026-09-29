# colorvid-quality-colormnet

Greffon **« quality »** de [ColorVid](https://github.com/TommyLec/colorvid) —
moteur de colorisation [ColorMNet](https://github.com/yyang181/colormnet)
(ECCV 2024 — Yang, Dong, Tang, Pan).

## Pourquoi un dépôt séparé

Ce module est une **œuvre dérivée** de ColorMNet : notre code réimplémente son
architecture, les fixtures de test ont été calculées par son code amont, et les
graphes ONNX exportés en descendent. Le **partage à l'identique** de la
CC BY-NC-SA 4.0 s'applique donc, et il ne peut pas être redistribué sous la licence
permissive du cœur.

Séparer les dépôts est la façon la plus propre de tenir cette frontière : **une
frontière de licence devient une frontière de dépôt**. Le cœur (Apache-2.0) ne
dépend jamais d'ici ; c'est l'inverse qui est vrai, et dans un seul sens.

## Licence — à lire avant tout usage

**CC BY-NC-SA 4.0** (texte intégral dans [`LICENSE`](LICENSE)) :

| | |
| :--- | :--- |
| **BY** | créditer ColorMNet (Yang, Dong, Tang, Pan — ECCV 2024), lier la source et la licence, et **indiquer les modifications** |
| **NC** | **usage non commercial** — un binaire produit avec ce moteur est un binaire non commercial |
| **SA** | toute adaptation reste sous CC BY-NC-SA 4.0 |

ColorMNet dérive lui-même de [XMem](https://github.com/hkchengrex/XMem) (MIT) et
emprunte à [DINOv2](https://github.com/facebookresearch/dinov2) (Apache-2.0) : ces
œuvres sont créditées dans `src/catalog.rs` et dans le `NOTICE.txt` écrit à côté des
poids téléchargés.

Toute contribution ici est acceptée sous cette même licence.

## Poids

**Jamais versionnés, jamais distribués.** 8 fichiers, 477 Mo, téléchargés au premier
usage et vérifiés par empreinte SHA-256 (`assets/weights.json` porte les empreintes,
les tailles et la provenance). Le texte de licence et un `NOTICE.txt` sont écrits
à côté des fichiers téléchargés.

L'URL d'hébergement reste à renseigner (`base_url` dans `assets/weights.json`) ; en
attendant, un dossier de poids existant peut être indiqué par la variable
`COLORVID_COLORMNET_GRAPHS`.

## Utilisation

Ce greffon est consommé par le cœur de ColorVid :

```bash
git clone https://github.com/TommyLec/colorvid
cd colorvid/app
cargo build --features quality-colormnet
```

Le cœur ajoute alors une entrée au catalogue des moteurs, la licence et
l'attribution étant transmises par `src/catalog.rs` — l'interface n'a besoin de
connaître aucun nom de moteur.

## Structure

```
src/
  engine.rs     orchestration des quatre graphes ONNX + état récurrent
  memory.rs     mémoire à trois niveaux (recherche, top-k, readout)
  store.rs      magasin de clés/valeurs
  catalog.rs    identifiant, licence, attribution, résolution des graphes
  download.rs   téléchargement vérifié par empreinte + notices
  geometry.rs   mise à l'échelle et découpage
  input.rs      préparation des entrées
  state.rs      état récurrent entre frames
assets/         manifeste des poids (empreintes, tailles, provenance)
tests/          fixtures calculées par le code amont + test d'orchestration
examples/       outil d'évaluation A/B dans le pipeline réel
```
