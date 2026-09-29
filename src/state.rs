//! État récurrent de ColorMNet — **externalisé** hors du graphe ONNX.
//!
//! ColorMNet est un modèle à mémoire (dérivé de XMem) : l'inférence n'est pas une
//! fonction pure frame → `ab`, mais une récurrence. Les graphes ONNX exportés
//! (jalon C2) sont **sans état** ; c'est ici que vivent la mémoire clé/valeur
//! (déléguée à [`crate::store::MemoryStore`]), la mémoire vive (`hidden`) et les
//! deux derniers tenseurs clé/valeur utilisés par l'attention court terme.
//!
//! Le calendrier d'exécution (quand mémoriser, quand segmenter) est reproduit à
//! l'identique de `inference/inference_core.InferenceCore.step` et **vérifié par
//! test** contre une exécution PyTorch réelle (`reference_phase_schedule`).
//!
//! La **rétention** de la mémoire est une politique explicite
//! ([`MidTermPolicy`]) : l'amont consolide les frames anciennes en prototypes
//! (mémoire long terme), le portage Rust les oublie ou les garde toutes selon
//! l'arbitrage mesuré (cf. `docs/plan-colormnet-quality.md`).

use crate::store::{MemoryStore, StoreError};

/// Rétention de la mémoire moyen terme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MidTermPolicy {
    /// Fenêtre glissante : au-delà de `frames` pas mémorisés, les frames les
    /// plus anciennes sont **oubliées** (aucune consolidation en prototypes).
    /// C'est le repli « RAM bornée d'abord » de `AGENTS.md`.
    Window { frames: usize },
    /// Aucune éviction : la mémoire grandit avec le plan. Réservé à
    /// l'évaluation (coût de calcul et RAM linéaires en durée de plan).
    Unbounded,
}

impl Default for MidTermPolicy {
    fn default() -> Self {
        Self::Window { frames: 10 }
    }
}

/// Paramètres de récurrence (mêmes noms que la configuration amont).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineConfig {
    /// `mem_every` : une frame est mémorisée tous les N pas.
    pub mem_every: usize,
    /// `deep_update_every` : négatif = synchronisé sur la mémorisation.
    pub deep_update_every: i64,
    /// Rétention de la mémoire moyen terme.
    pub mid_term: MidTermPolicy,
    /// `top_k` : nombre de clés retenues par la normalisation de la lecture
    /// mémoire (`do_softmax`). Borne aussi le coût de l'attention.
    pub top_k: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            mem_every: 5,
            deep_update_every: -1,
            mid_term: MidTermPolicy::default(),
            top_k: 30,
        }
    }
}

impl EngineConfig {
    /// `deep_update_every < 0` : la mise à jour profonde suit la mémorisation.
    pub fn deep_update_sync(&self) -> bool {
        self.deep_update_every < 0
    }

    /// Taille de la fenêtre mémoire, si bornée.
    pub fn window_frames(&self) -> Option<usize> {
        match self.mid_term {
            MidTermPolicy::Window { frames } => Some(frames),
            MidTermPolicy::Unbounded => None,
        }
    }
}

/// Ce que le pas courant doit faire — calculé AVANT l'exécution des graphes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepPhase {
    /// Indice du pas (0 = première frame du plan).
    pub step: usize,
    /// Une image-exemplaire est fournie à ce pas (première frame).
    pub has_exemplar: bool,
    /// Le réseau doit segmenter (prédire les canaux `ab`).
    pub needs_segment: bool,
    /// La frame entre en mémoire.
    pub is_mem_frame: bool,
    /// Mise à jour profonde de la mémoire vive.
    pub is_deep_update: bool,
    /// Mise à jour normale de la mémoire vive.
    pub is_normal_update: bool,
}

/// Ce qu'un pas apporte à la mémoire : les tenseurs produits par les graphes.
///
/// * `key` : `[ck, hw]` ;
/// * `shrinkage` : `[hw]` ;
/// * `value` : `[num_objects, cv, hw]` ;
/// * `objects` : étiquettes **1-based** (`all_labels` amont, `[1, 2]` en
///   colorisation : les deux canaux de chroma).
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryFrame {
    pub objects: Vec<usize>,
    pub key: Vec<f32>,
    pub shrinkage: Vec<f32>,
    pub value: Vec<f32>,
}

/// État complet d'une séquence (un plan). `reset()` à chaque coupe.
#[derive(Debug, Clone, Default)]
pub struct ColorMNetState {
    step: usize,
    last_mem_step: usize,
    last_deep_update_step: i64,
    last_key: Option<Vec<f32>>,
    last_value: Option<Vec<f32>>,
    hidden: Option<Vec<f32>>,
    memory: MemoryStore,
    /// Pas des frames mémorisées, dans l'ordre d'insertion (diagnostic/tests).
    mem_steps: Vec<usize>,
}

impl ColorMNetState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Remise à zéro complète — **obligatoire à chaque changement de plan**
    /// (la mémoire d'un plan ne doit pas déborder sur le suivant).
    pub fn reset(&mut self) {
        let cfg = EngineConfig::default();
        self.reset_with(&cfg);
    }

    /// Remise à zéro en tenant compte de la configuration (l'état initial de
    /// `last_deep_update_step` dépend de `deep_update_every`).
    pub fn reset_with(&mut self, config: &EngineConfig) {
        self.step = 0;
        self.last_mem_step = 0;
        self.last_deep_update_step = if config.deep_update_sync() {
            0
        } else {
            -config.deep_update_every
        };
        self.last_key = None;
        self.last_value = None;
        self.hidden = None;
        self.memory = MemoryStore::new();
        self.mem_steps.clear();
    }

    pub fn step(&self) -> usize {
        self.step
    }

    pub fn hidden(&self) -> Option<&[f32]> {
        self.hidden.as_deref()
    }

    pub fn last_key(&self) -> Option<&[f32]> {
        self.last_key.as_deref()
    }

    pub fn last_value(&self) -> Option<&[f32]> {
        self.last_value.as_deref()
    }

    /// Mémoire clé/valeur en lecture seule (diagnostic de l'orchestration).
    pub fn memory(&self) -> &MemoryStore {
        &self.memory
    }

    /// Nombre de frames **encore** en mémoire.
    pub fn mid_term_len(&self) -> usize {
        match self.memory.frame_hw() {
            0 => 0,
            hw => self.memory.size() / hw,
        }
    }

    /// Pas des frames encore en mémoire, dans l'ordre.
    pub fn mid_term_steps(&self) -> Vec<usize> {
        let live = self.mid_term_len();
        self.mem_steps[self.mem_steps.len().saturating_sub(live)..].to_vec()
    }

    /// Calendrier du pas courant (formules identiques à `InferenceCore.step`).
    ///
    /// `has_exemplar` : la frame-exemplaire est fournie ; `is_last` : dernière
    /// frame du plan (`end=True` dans l'amont, qui neutralise mémorisation et
    /// mises à jour).
    pub fn phase(&self, config: &EngineConfig, has_exemplar: bool, is_last: bool) -> StepPhase {
        let step = self.step;
        let need_segment = step > 0;
        let is_mem_frame =
            ((step - self.last_mem_step >= config.mem_every) || has_exemplar) && !is_last;
        let is_deep_update = (if config.deep_update_sync() {
            is_mem_frame
        } else {
            step as i64 - self.last_deep_update_step >= config.deep_update_every
        }) && !is_last;
        let is_normal_update = (!config.deep_update_sync() || !is_deep_update) && !is_last;
        StepPhase {
            step,
            has_exemplar,
            // NB : `end` ne neutralise PAS la segmentation — la dernière frame doit
            // produire une sortie, elle n'alimente simplement pas la mémoire.
            needs_segment: need_segment,
            is_mem_frame,
            is_deep_update,
            is_normal_update,
        }
    }

    /// Clôt le pas courant : mémorise la frame si le calendrier le demande,
    /// puis avance le calendrier.
    ///
    /// **À appeler à chaque frame**, avec `Some(frame)` exactement quand
    /// `phase.is_mem_frame` — d'où la vérification croisée. Un compteur de pas
    /// qui n'avance qu'aux mémorisations désynchronise tout le calendrier :
    /// c'est le bug qu'a attrapé la comparaison pas à pas avec PyTorch (§13.7).
    pub fn commit(
        &mut self,
        config: &EngineConfig,
        phase: StepPhase,
        frame: Option<&MemoryFrame>,
    ) -> Result<(), StoreError> {
        match (phase.is_mem_frame, frame) {
            (true, Some(frame)) => {
                self.memory
                    .add(&frame.key, &frame.shrinkage, &frame.value, &frame.objects)?;
                self.mem_steps.push(phase.step);
                self.last_mem_step = phase.step;
                self.last_key = Some(frame.key.clone());
                self.last_value = Some(frame.value.clone());
                if phase.is_deep_update {
                    self.last_deep_update_step = phase.step as i64;
                }
                self.enforce_retention(config)?;
            }
            (false, None) => {}
            _ => {
                return Err(StoreError::Shape(
                    "incohérence de pas : un pas mémorisé doit fournir ses tenseurs, un pas \
                     non mémorisé ne doit pas en fournir"
                        .into(),
                ))
            }
        }
        self.step += 1;
        Ok(())
    }

    /// Applique la politique de rétention après une insertion.
    fn enforce_retention(&mut self, config: &EngineConfig) -> Result<(), StoreError> {
        let Some(frames) = config.window_frames() else {
            return Ok(());
        };
        if frames == 0 || self.memory.frame_hw() == 0 {
            return Err(StoreError::Shape(
                "fenêtre mémoire nulle : la mémoire serait vidée à chaque pas".into(),
            ));
        }
        let keep = frames * self.memory.frame_hw();
        while self.memory.size() > keep {
            self.memory
                .drop_oldest_frames(1)
                .map_err(|_| StoreError::Shape("éviction impossible".into()))?;
        }
        Ok(())
    }

    /// Renseigne la mémoire vive produite par une segmentation.
    pub fn set_hidden(&mut self, hidden: Vec<f32>) {
        self.hidden = Some(hidden);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Calendrier de référence **mesuré** sur l'implémentation PyTorch
    /// (14 pas, `mem_every=5`, `deep_update_every=-1`, exemplaire au pas 0,
    /// dernière frame marquée `end`) : (step, exemplaire, segment, mem, deep, normal).
    const REFERENCE_PHASE_SCHEDULE: [(usize, bool, bool, bool, bool, bool); 14] = [
        (0, true, false, true, true, false),
        (1, false, true, false, false, true),
        (2, false, true, false, false, true),
        (3, false, true, false, false, true),
        (4, false, true, false, false, true),
        (5, false, true, true, true, false),
        (6, false, true, false, false, true),
        (7, false, true, false, false, true),
        (8, false, true, false, false, true),
        (9, false, true, false, false, true),
        (10, false, true, true, true, false),
        (11, false, true, false, false, true),
        (12, false, true, false, false, true),
        (13, false, true, false, false, false),
    ];

    /// Joue un pas complet (calendrier + mémorisation) comme le fait le moteur.
    fn step(state: &mut ColorMNetState, config: &EngineConfig, has_exemplar: bool, is_last: bool, seed: f32) {
        let phase = state.phase(config, has_exemplar, is_last);
        let frame = frame(seed);
        state
            .commit(config, phase, phase.is_mem_frame.then_some(&frame))
            .expect("pas cohérent");
    }

    /// Frame minimale cohérente : ck=2, hw=3, cv=2, deux objets.
    fn frame(seed: f32) -> MemoryFrame {
        MemoryFrame {
            objects: vec![1, 2],
            key: (0..6).map(|i| seed + i as f32).collect(),
            shrinkage: (0..3).map(|i| seed + i as f32).collect(),
            value: (0..12).map(|i| seed + i as f32).collect(),
        }
    }

    #[test]
    fn reference_phase_schedule_is_reproduced() {
        let config = EngineConfig::default();
        let mut state = ColorMNetState::new();
        state.reset_with(&config);
        let mut phases = Vec::new();
        for (i, (step, exemplar, segment, mem, deep, normal)) in
            REFERENCE_PHASE_SCHEDULE.iter().enumerate()
        {
            let is_last = i == REFERENCE_PHASE_SCHEDULE.len() - 1;
            let phase = state.phase(&config, *exemplar, is_last);
            assert_eq!(phase.step, *step, "indice du pas {i}");
            assert_eq!(phase.needs_segment, *segment, "needs_segment au pas {i}");
            assert_eq!(phase.is_mem_frame, *mem, "is_mem_frame au pas {i}");
            assert_eq!(phase.is_deep_update, *deep, "is_deep_update au pas {i}");
            assert_eq!(phase.is_normal_update, *normal, "is_normal_update au pas {i}");
            phases.push(phase);
            let frame = frame(i as f32);
            state
                .commit(&config, phase, phase.is_mem_frame.then_some(&frame))
                .unwrap();
            if !is_last {
                state.set_hidden(vec![1.0]);
            }
        }
        // les frames mémorisées sont celles du calendrier, dans l'ordre
        assert_eq!(state.mid_term_steps(), vec![0, 5, 10]);
        assert_eq!(state.memory().num_groups(), 1);
        assert_eq!(state.memory().v_size(0), state.memory().size());
    }

    #[test]
    fn ring_buffer_evicts_the_oldest_frames() {
        let config = EngineConfig {
            mem_every: 1,
            mid_term: MidTermPolicy::Window { frames: 3 },
            ..EngineConfig::default()
        };
        let mut state = ColorMNetState::new();
        state.reset_with(&config);
        for i in 0..8 {
            step(&mut state, &config, i == 0, false, i as f32);
        }
        assert_eq!(state.mid_term_len(), 3);
        assert_eq!(state.mid_term_steps(), vec![5, 6, 7]);
        assert_eq!(state.memory().size(), 9);
        // la dernière clé mémorisée est celle du pas 7
        assert_eq!(state.last_key(), Some(frame(7.0).key.as_slice()));
        assert_eq!(state.step(), 8);
    }

    #[test]
    fn an_unbounded_policy_keeps_every_frame() {
        let config = EngineConfig {
            mem_every: 1,
            mid_term: MidTermPolicy::Unbounded,
            ..EngineConfig::default()
        };
        let mut state = ColorMNetState::new();
        state.reset_with(&config);
        for i in 0..30 {
            step(&mut state, &config, i == 0, false, i as f32);
        }
        assert_eq!(state.mid_term_len(), 30);
    }

    #[test]
    fn an_inconsistent_frame_is_reported_and_does_not_advance_the_calendar() {
        let config = EngineConfig::default();
        let mut state = ColorMNetState::new();
        state.reset_with(&config);
        let phase = state.phase(&config, true, false);
        let mut broken = frame(0.0);
        broken.value.truncate(5); // valeur tronquée : dimensions incohérentes
        assert!(state.commit(&config, phase, Some(&broken)).is_err());
        assert_eq!(state.step(), 0, "le calendrier ne doit pas avancer");
        assert!(state.memory().is_empty());
    }

    #[test]
    fn reset_clears_everything_between_shots() {
        let config = EngineConfig::default();
        let mut state = ColorMNetState::new();
        state.reset_with(&config);
        for i in 0..3 {
            step(&mut state, &config, i == 0, false, i as f32);
        }
        state.set_hidden(vec![3.0]);
        assert!(state.mid_term_len() > 0 && state.hidden().is_some());

        state.reset_with(&config);
        assert_eq!(state.step(), 0);
        assert_eq!(state.mid_term_len(), 0);
        assert!(state.mid_term_steps().is_empty());
        assert!(state.hidden().is_none());
        assert!(state.last_key().is_none());
        assert!(state.last_value().is_none());
        assert!(state.memory().is_empty());
        // et la séquence repart comme une première frame de plan
        let phase = state.phase(&config, true, false);
        assert!(phase.has_exemplar && phase.is_mem_frame && !phase.needs_segment);
    }

    #[test]
    fn last_frame_of_a_shot_freezes_memory_updates() {
        let config = EngineConfig::default();
        let mut state = ColorMNetState::new();
        state.reset_with(&config);
        // amener l'état au pas 3 (pas de la fin d'un plan court)
        for i in 0..3 {
            step(&mut state, &config, i == 0, false, i as f32);
        }
        let phase = state.phase(&config, false, true);
        // la dernière frame n'alimente plus la mémoire…
        assert!(!phase.is_mem_frame && !phase.is_deep_update && !phase.is_normal_update);
        // …mais elle est bien segmentée (sa sortie doit être produite).
        assert!(phase.needs_segment);
    }
}
