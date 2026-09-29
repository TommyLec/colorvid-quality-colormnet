//! Mémoire clé/valeur et **suivi par objet** — portée de
//! `inference/kv_memory_store.py` (`KeyValueMemoryStore`) et de
//! `inference/memory_manager.MemoryManager.match_memory`.
//!
//! ColorMNet propage les **deux canaux de chroma comme deux « objets »** : les
//! clés sont communes, mais les valeurs sont stockées **par groupe d'objets**
//! (`objects` = `all_labels` = `[1, 2]` amont, `1` = fond exclu). Un groupe est
//! créé quand de nouveaux objets apparaissent ; les objets d'un même groupe
//! partagent la même étendue temporelle.
//!
//! Conséquence pratique de notre configuration, **mesurée** (`c4_probe.py`,
//! `tests/fixtures/memory_store.json`) : les deux canaux sont fournis dès le
//! premier pas, donc il n'existe **qu'un seul groupe** et `v_size(0) == size`
//! à chaque pas. Le code reste néanmoins général (et testé sur un cas
//! multi-groupe) : c'est peu de code, et cela évite un faux silencieux si
//! `all_labels` changeait un jour.
//!
//! Disposition mémoire : clés et sélection en `[ck, n]` (`idx = c*n + i`),
//! valeurs d'un groupe en `[num_obj, cv, n_g]` (`idx = o*cv*n_g + c*n_g + i`),
//! avec `n_g` = nombre de colonnes **valides pour ce groupe**. Pour un groupe
//! `gi > 0`, ces `n_g` colonnes correspondent aux **`n_g` dernières** colonnes
//! de clés (l'amont : `similarity[:, -v_size(gi):]`).

use crate::memory::{readout, similarity, top_k_softmax};

/// Erreurs de la mémoire clé/valeur.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum StoreError {
    #[error("étiquette d'objet {0} invalide (0 = fond, jamais stocké)")]
    InvalidObjectLabel(usize),
    #[error("objet {0} suivi en mémoire mais absent du pas courant : suivi incohérent")]
    TrackedObjectMissing(usize),
    #[error("dimensions incohérentes : {0}")]
    Shape(String),
    #[error("objets insérés dans le désordre (l'amont exige un ordre trié)")]
    UnsortedObjects,
    #[error("mémoire vide : lecture impossible")]
    Empty,
}

/// Un groupe d'objets : mêmes colonnes de clés, valeurs propres au groupe.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectGroup {
    /// Indices d'objets **0-based** couverts par ce groupe, dans l'ordre des
    /// lignes de `values`.
    objects: Vec<usize>,
    /// Valeurs `[num_obj, cv, n]`.
    values: Vec<f32>,
    /// Nombre de colonnes (frames) valides pour ce groupe.
    n: usize,
}

impl ObjectGroup {
    pub fn objects(&self) -> &[usize] {
        &self.objects
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    /// Colonnes valides pour ce groupe.
    pub fn n(&self) -> usize {
        self.n
    }
}

/// Lecture mémoire d'un pas : `[num_objects, cv, hw]` (objets concaténés dans
/// l'ordre des groupes, comme `torch.cat([...], 0)` en amont).
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryReadout {
    pub data: Vec<f32>,
    pub num_objects: usize,
    pub cv: usize,
    pub hw: usize,
}

impl MemoryReadout {
    /// Vue `[cv, hw]` d'un objet donné.
    pub fn object(&self, index: usize) -> &[f32] {
        let per_object = self.cv * self.hw;
        &self.data[index * per_object..(index + 1) * per_object]
    }
}

/// Mémoire moyen terme : clés partagées + valeurs par groupe d'objets.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MemoryStore {
    ck: usize,
    cv: usize,
    frame_hw: usize,
    /// `[ck, n]`
    keys: Vec<f32>,
    /// `[n]`
    shrinkage: Vec<f32>,
    groups: Vec<ObjectGroup>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Nombre de colonnes de clés (une frame = `frame_hw` colonnes).
    pub fn size(&self) -> usize {
        self.shrinkage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    pub fn ck(&self) -> usize {
        self.ck
    }

    pub fn cv(&self) -> usize {
        self.cv
    }

    pub fn frame_hw(&self) -> usize {
        self.frame_hw
    }

    /// Nombre total d'objets suivis (somme des groupes) — dimension de la
    /// lecture mémoire en amont (`all_readout_mem.shape[0]`).
    pub fn total_objects(&self) -> usize {
        self.groups.iter().map(ObjectGroup::len).sum()
    }

    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }

    pub fn groups(&self) -> &[ObjectGroup] {
        &self.groups
    }

    /// `get_v_size(gi)` — colonnes valides du groupe `gi`.
    pub fn v_size(&self, gi: usize) -> usize {
        self.groups[gi].n
    }

    pub fn obj_groups(&self) -> Vec<Vec<usize>> {
        self.groups.iter().map(|g| g.objects.clone()).collect()
    }

    pub fn keys(&self) -> &[f32] {
        &self.keys
    }

    pub fn shrinkage(&self) -> &[f32] {
        &self.shrinkage
    }

    /// Ajoute une frame en mémoire (`KeyValueMemoryStore.add`).
    ///
    /// * `key` : `[ck, hw]` ; `shrinkage` : `[hw]` ;
    /// * `value` : `[num_objects, cv, hw]` (une ligne par objet, dans l'ordre
    ///   des étiquettes de `objects`) ;
    /// * `objects` : étiquettes **1-based** (`all_labels` amont, `[1, 2]`).
    ///
    /// Aucune écriture n'a lieu si la validation échoue : l'état reste intact.
    ///
    /// NB : la **sélection mémoire** de l'amont (`KeyValueMemoryStore.e`) n'est
    /// pas portée — elle n'alimente que la consolidation en prototypes, écartée
    /// sur mesure (§13.3 du plan). La sélection de *requête*, elle, est bien
    /// utilisée par [`MemoryStore::match_memory`].
    pub fn add(
        &mut self,
        key: &[f32],
        shrinkage: &[f32],
        value: &[f32],
        objects: &[usize],
    ) -> Result<(), StoreError> {
        let hw = shrinkage.len();
        if hw == 0 {
            return Err(StoreError::Shape("frame sans colonne".into()));
        }
        if objects.is_empty() {
            return Err(StoreError::Shape("aucun objet pour cette frame".into()));
        }

        // La première frame fixe les dimensions ; rien n'est écrit avant que
        // toutes les validations soient passées.
        let first = self.is_empty();
        let (ck, cv) = if first {
            if !key.len().is_multiple_of(hw) {
                return Err(StoreError::Shape(format!(
                    "clé de longueur {} non divisible par {hw}",
                    key.len()
                )));
            }
            let per_object = hw * objects.len();
            if value.is_empty() || !value.len().is_multiple_of(per_object) {
                return Err(StoreError::Shape(format!(
                    "valeur de longueur {} non divisible par {per_object}",
                    value.len()
                )));
            }
            (key.len() / hw, value.len() / per_object)
        } else {
            if hw != self.frame_hw {
                return Err(StoreError::Shape(format!(
                    "frame de {hw} colonnes, {} attendues",
                    self.frame_hw
                )));
            }
            (self.ck, self.cv)
        };

        let n = self.size();
        let expected = ck * hw;
        if key.len() != expected {
            return Err(StoreError::Shape(format!(
                "clé {} attendu {expected}",
                key.len()
            )));
        }
        let per_object = cv * hw;
        if value.len() != objects.len() * per_object {
            return Err(StoreError::Shape(format!(
                "valeur {} attendu {}",
                value.len(),
                objects.len() * per_object
            )));
        }

        // Objets à placer : l'amont décale de 1 (le fond n'a pas de valeur).
        let mut remaining: Vec<usize> = Vec::with_capacity(objects.len());
        for &obj in objects {
            if obj == 0 {
                return Err(StoreError::InvalidObjectLabel(0));
            }
            remaining.push(obj - 1);
        }
        // Les objets déjà suivis par un groupe doivent être annoncés à nouveau.
        for group in &self.groups {
            for &obj in &group.objects {
                match remaining.iter().position(|&x| x == obj) {
                    Some(pos) => {
                        remaining.remove(pos);
                    }
                    None => return Err(StoreError::TrackedObjectMissing(obj + 1)),
                }
            }
        }
        // Ordre trié imposé par l'amont (`all_objects` trié).
        let mut all: Vec<usize> = self
            .groups
            .iter()
            .flat_map(|g| g.objects.iter().copied())
            .collect();
        all.extend(remaining.iter().copied());
        if !all.windows(2).all(|w| w[0] < w[1]) {
            return Err(StoreError::UnsortedObjects);
        }

        // --- validation passée : on écrit ---
        if first {
            self.ck = ck;
            self.cv = cv;
            self.frame_hw = hw;
        }
        // Colonnes de clés : `[ck, n]` → `[ck, n + hw]`.
        if n == 0 {
            self.keys = key.to_vec();
        } else {
            append_columns(&mut self.keys, key, ck, n, hw);
        }
        self.shrinkage.extend_from_slice(shrinkage);

        // Valeurs : chaque groupe déjà su reçoit les colonnes de ses objets.
        for group in &mut self.groups {
            let old_n = group.n;
            let mut values = Vec::with_capacity(group.len() * cv * (old_n + hw));
            for (slot, &obj) in group.objects.iter().enumerate() {
                let old = &group.values[slot * cv * old_n..(slot + 1) * cv * old_n];
                let new = &value[obj * per_object..(obj + 1) * per_object];
                for (o, s) in old.chunks_exact(old_n).zip(new.chunks_exact(hw)) {
                    values.extend_from_slice(o);
                    values.extend_from_slice(s);
                }
            }
            group.values = values;
            group.n = old_n + hw;
        }

        // Objets restants : nouveau groupe, avec les seules colonnes du pas.
        if !remaining.is_empty() {
            let mut values = Vec::with_capacity(remaining.len() * per_object);
            for &obj in &remaining {
                values.extend_from_slice(&value[obj * per_object..(obj + 1) * per_object]);
            }
            self.groups.push(ObjectGroup {
                objects: remaining,
                values,
                n: hw,
            });
        }
        Ok(())
    }

    /// `MemoryManager.match_memory` **sans mémoire long terme**.
    ///
    /// L'affinité est calculée **groupe par groupe** (chaque groupe ne voit que
    /// ses propres colonnes), puis les lectures sont concaténées.
    pub fn match_memory(
        &self,
        query_key: &[f32],
        query_selection: Option<&[f32]>,
        top_k: usize,
    ) -> Result<MemoryReadout, StoreError> {
        if self.is_empty() {
            return Err(StoreError::Empty);
        }
        if query_key.is_empty() || !query_key.len().is_multiple_of(self.ck) {
            return Err(StoreError::Shape(format!(
                "clé de requête {} non divisible par ck={}",
                query_key.len(),
                self.ck
            )));
        }
        let hw = query_key.len() / self.ck;
        let n = self.size();
        let sim = similarity(
            &self.keys,
            &self.shrinkage,
            query_key,
            query_selection,
            self.ck,
            n,
            hw,
        );

        let mut data = vec![0.0f32; self.total_objects() * self.cv * hw];
        let mut first_object = 0usize;
        for group in &self.groups {
            let n_g = group.n;
            if n_g > n {
                return Err(StoreError::Shape(format!(
                    "groupe de {n_g} colonnes pour {n} clés"
                )));
            }
            // Le groupe ne voit que ses colonnes : les `n_g` dernières.
            let aff = top_k_softmax(&sim[(n - n_g) * hw..], n_g, hw, top_k);
            for (slot, _) in group.objects.iter().enumerate() {
                let values = &group.values[slot * self.cv * n_g..(slot + 1) * self.cv * n_g];
                let one = readout(&aff, values, self.cv, n_g, hw);
                let start = (first_object + slot) * self.cv * hw;
                data[start..start + self.cv * hw].copy_from_slice(&one);
            }
            first_object += group.len();
        }
        Ok(MemoryReadout {
            data,
            num_objects: self.total_objects(),
            cv: self.cv,
            hw,
        })
    }

    /// Évince les `frames` frames les plus anciennes (`sieve_by_range`).
    ///
    /// Repli « RAM bornée d'abord » : au lieu de consolider les frames anciennes
    /// en prototypes (mémoire long terme), on les oublie. `frames` est un nombre
    /// de **frames**, converti en colonnes via `frame_hw`.
    pub fn drop_oldest_frames(&mut self, frames: usize) -> Result<usize, StoreError> {
        let n = self.size();
        let drop = frames * self.frame_hw;
        if drop == 0 || n == 0 {
            return Ok(0);
        }
        if drop >= n {
            return Err(StoreError::Shape(format!(
                "éviction de {drop} colonnes sur {n} : la mémoire serait vide"
            )));
        }
        let keep = n - drop;

        let mut keys = Vec::with_capacity(self.ck * keep);
        for chunk in self.keys.chunks_exact(n).map(|c| &c[drop..]) {
            keys.extend_from_slice(chunk);
        }
        self.keys = keys;
        self.shrinkage.drain(..drop);

        // Un groupe plus court que la fenêtre conservée n'est pas concerné : ses
        // colonnes sont toutes dans la fenêtre (comportement amont : `min_size`).
        for group in &mut self.groups {
            let old_n = group.n;
            let lost = drop.saturating_sub(n - old_n);
            if lost == 0 {
                continue;
            }
            let new_n = old_n - lost;
            let mut values = Vec::with_capacity(group.len() * self.cv * new_n);
            for slot in 0..group.len() {
                let base = slot * self.cv * old_n;
                for c in 0..self.cv {
                    let start = base + c * old_n + lost;
                    values.extend_from_slice(&group.values[start..start + new_n]);
                }
            }
            group.values = values;
            group.n = new_n;
        }
        Ok(drop)
    }
}

/// Ajoute des colonnes `[ck, hw]` à un tampon `[ck, n]` (disposition par canal).
fn append_columns(dst: &mut Vec<f32>, src: &[f32], ck: usize, n: usize, hw: usize) {
    let mut out = Vec::with_capacity(ck * (n + hw));
    for (old, new) in dst.chunks_exact(n).zip(src.chunks_exact(hw)) {
        out.extend_from_slice(old);
        out.extend_from_slice(new);
    }
    *dst = out;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// Fixture **calculée par le code amont** : c'est la vérité de test.
    fn fixture(raw: &str) -> Value {
        serde_json::from_str(raw).expect("fixture JSON")
    }

    fn num(f: &Value, key: &str) -> usize {
        f[key].as_u64().unwrap_or_else(|| panic!("{key}")) as usize
    }

    fn f32s(f: &Value, key: &str) -> Vec<f32> {
        f[key]
            .as_array()
            .unwrap_or_else(|| panic!("{key}"))
            .iter()
            .map(|x| x.as_f64().expect("nombre") as f32)
            .collect()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// Colonnes identiques pour un groupe : `n` frames de `hw` colonnes.
    fn frame(seed: f32, len: usize) -> Vec<f32> {
        (0..len).map(|i| seed + i as f32 * 0.25).collect()
    }

    fn store_with(objects: &[usize]) -> MemoryStore {
        let mut store = MemoryStore::new();
        let ck = 2;
        let hw = 3;
        for step in 0..2 {
            let s = step as f32;
            store
                .add(
                    &frame(s, ck * hw),
                    &frame(s, hw),
                    &frame(s, objects.len() * 2 * hw),
                    objects,
                )
                .expect("ajout");
        }
        store
    }

    #[test]
    fn both_chroma_channels_share_a_single_group() {
        // Configuration du produit : `all_labels = [1, 2]` à chaque pas.
        let store = store_with(&[1, 2]);
        assert_eq!(store.num_groups(), 1);
        assert_eq!(store.obj_groups(), vec![vec![0, 1]]);
        assert_eq!(store.size(), 6);
        assert_eq!(store.v_size(0), store.size());
        assert_eq!(store.total_objects(), 2);
        assert_eq!(store.cv(), 2);
        assert_eq!(store.ck(), 2);
    }

    #[test]
    fn a_later_object_opens_a_second_group_with_its_own_extent() {
        let mut store = store_with(&[1, 2]);
        // troisième objet au pas suivant : seul, dans son propre groupe
        store
            .add(
                &frame(9.0, 2 * 3),
                &frame(9.0, 3),
                &frame(9.0, 3 * 2 * 3),
                &[1, 2, 3],
            )
            .expect("ajout");
        assert_eq!(store.num_groups(), 2);
        assert_eq!(store.obj_groups(), vec![vec![0, 1], vec![2]]);
        assert_eq!(store.size(), 9);
        assert_eq!(store.v_size(0), 9, "le groupe 0 voit toutes les colonnes");
        assert_eq!(store.v_size(1), 3, "le groupe 1 n'a que ses colonnes");
    }

    #[test]
    fn dropping_a_tracked_object_is_rejected_without_mutating() {
        let mut store = store_with(&[1, 2]);
        let before = store.clone();
        let err = store
            .add(&frame(1.0, 6), &frame(1.0, 3), &frame(1.0, 6), &[1])
            .unwrap_err();
        assert_eq!(err, StoreError::TrackedObjectMissing(2));
        assert_eq!(store, before, "aucune écriture partielle");
    }

    #[test]
    fn shape_mismatch_is_rejected() {
        let mut store = store_with(&[1, 2]);
        let err = store
            .add(&frame(1.0, 6), &frame(1.0, 3), &frame(1.0, 7), &[1, 2])
            .unwrap_err();
        assert!(matches!(err, StoreError::Shape(_)), "{err}");
    }

    #[test]
    fn eviction_drops_the_oldest_frames_and_keeps_keys_aligned() {
        let mut store = store_with(&[1, 2]);
        assert_eq!(store.drop_oldest_frames(1).unwrap(), 3);
        assert_eq!(store.size(), 3);
        assert_eq!(store.v_size(0), 3);
        // la clé restante est bien celle du second pas
        assert_eq!(store.keys(), frame(1.0, 6).as_slice());
        assert_eq!(store.shrinkage(), frame(1.0, 3).as_slice());
        // évincer toute la mémoire est une erreur, pas un état vide silencieux
        assert!(store.drop_oldest_frames(1).is_err());
    }

    #[test]
    fn readout_before_any_frame_is_an_explicit_error() {
        let store = MemoryStore::new();
        let err = store.match_memory(&[0.0; 6], None, 3).unwrap_err();
        assert_eq!(err, StoreError::Empty);
    }

    /// Cas de référence **calculé par le code amont** (`MemoryManager` réel,
    /// `enable_long_term=False`) sur une séquence à **deux groupes** : le groupe
    /// 0 (objets 1-2) présent depuis le premier pas, le groupe 1 (objet 3) entré
    /// au troisième pas.
    #[test]
    fn matches_the_upstream_multi_group_reference() {
        let f = fixture(include_str!("../tests/fixtures/memory_store.json"));
        let ck = num(&f, "ck");
        let cv = num(&f, "cv");
        let hw = num(&f, "hw");
        let top_k = num(&f, "top_k");

        let mut store = MemoryStore::new();
        for (i, add) in f["adds"].as_array().expect("adds").iter().enumerate() {
            let objects: Vec<usize> = add["objects"]
                .as_array()
                .expect("objects")
                .iter()
                .map(|x| x.as_u64().expect("étiquette") as usize)
                .collect();
            store
                .add(
                    &f32s(&f, &format!("key{i}")),
                    &f32s(&f, &format!("shrinkage{i}")),
                    &f32s(&f, &format!("value{i}")),
                    &objects,
                )
                .unwrap_or_else(|e| panic!("ajout {i} : {e}"));
            assert_eq!(
                store.num_groups(),
                num(&f, &format!("num_groups{i}")),
                "groupes après l'ajout {i}"
            );
            let sizes: Vec<usize> = (0..store.num_groups()).map(|g| store.v_size(g)).collect();
            let expected: Vec<usize> = f[format!("v_size{i}")]
                .as_array()
                .expect("v_size")
                .iter()
                .map(|x| x.as_u64().expect("taille") as usize)
                .collect();
            assert_eq!(sizes, expected, "étendues des groupes après l'ajout {i}");
        }
        assert_eq!(store.obj_groups(), vec![vec![0, 1], vec![2]]);
        assert_eq!(store.ck(), ck);
        assert_eq!(store.cv(), cv);
        assert_eq!(store.frame_hw(), hw);

        let out = store
            .match_memory(&f32s(&f, "query_key"), None, top_k)
            .expect("lecture mémoire");
        let expected = f32s(&f, "readout");
        assert_eq!(out.num_objects, num(&f, "num_objects"));
        assert_eq!(out.cv, cv);
        assert_eq!(out.hw, hw);
        assert!(max_abs_diff(&out.data, &expected) < 1e-5, "lecture divergente");
    }

    /// Même discipline pour l'**éviction** : la référence est produite par
    /// `KeyValueMemoryStore.sieve_by_range` réel, pas par ma lecture du code.
    #[test]
    fn eviction_matches_the_upstream_sieve_by_range() {
        let f = fixture(include_str!("../tests/fixtures/memory_store_window.json"));
        let hw = num(&f, "hw");
        let top_k = num(&f, "top_k");

        let mut store = MemoryStore::new();
        for i in 0..num(&f, "drop_frames") + num(&f, "keep_frames") {
            store
                .add(
                    &f32s(&f, &format!("key{i}")),
                    &f32s(&f, &format!("shrinkage{i}")),
                    &f32s(&f, &format!("value{i}")),
                    &[1, 2],
                )
                .unwrap_or_else(|e| panic!("ajout {i} : {e}"));
        }
        let before = store.size();
        assert_eq!(before, (num(&f, "drop_frames") + num(&f, "keep_frames")) * hw);

        let dropped = store.drop_oldest_frames(num(&f, "drop_frames")).unwrap();
        assert_eq!(dropped, num(&f, "drop_frames") * hw);
        assert_eq!(store.size(), num(&f, "keep_frames") * hw);
        assert_eq!(store.v_size(0), store.size());

        let out = store
            .match_memory(&f32s(&f, "query_key"), None, top_k)
            .expect("lecture après éviction");
        let expected = f32s(&f, "readout");
        assert_eq!(out.num_objects, num(&f, "num_objects"));
        assert!(
            max_abs_diff(&out.data, &expected) < 1e-5,
            "lecture après éviction divergente"
        );
    }
}
