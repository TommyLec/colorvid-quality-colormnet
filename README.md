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

L'hébergement est protégé par des **URLs signées** (`base_url` dans
`assets/weights.json`). La signature est recalculée à chaque téléchargement — une
URL signée ne peut donc pas être stockée — et l'algorithme est celui du service :
MD5 de `secret + chemin + expiration`, en base64 URL-safe sans padding.

Le secret se fournit par variable d'environnement, ou à la compilation :

```bash
# à l'exécution, prioritaire
COLORVID_COLORMNET_SECRET=… ./colorvid

# ou figé dans un binaire de distribution (jamais dans ce dépôt)
COLORVID_COLORMNET_SECRET=… cargo build --release
```

Sans secret, les URLs sont demandées **non signées** : un service ouvert fonctionne
tel quel, un service protégé répond 403 avec un message qui le dit. Le secret ne doit
jamais être versionné.

> **Ce que cette signature protège, et ce qu'elle ne protège pas.** Un binaire
> distribué ne peut pas garder un secret : qui l'a peut l'extraire. Elle écarte donc
> le téléchargement automatisé et le lien partagé, **pas** un utilisateur déterminé.
> Si une protection réelle devient nécessaire, la voie est un point d'entrée public
> qui délivre des URLs signées à la demande, avec limitation de débit — l'application
> n'aurait alors aucun secret à porter.

Un dossier de poids local peut toujours être indiqué par
`COLORVID_COLORMNET_GRAPHS`, ce qui court-circuite tout téléchargement.

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
