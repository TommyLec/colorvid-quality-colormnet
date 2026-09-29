//! Orchestration des quatre graphes ONNX — le pas d'inférence de ColorMNet.
//!
//! Portage de `inference/inference_core.InferenceCore.step` : les graphes
//! exportés au C2 sont **sans état**, c'est ce module qui les enchaîne et qui
//! porte l'état entre deux frames.
//!
//! ```text
//! encode_key(image) ──► key, shrinkage, selection, f16, f8, f4
//!        │
//!        ├─ si need_segment :
//!        │     match_memory(key, selection)        (crate::store)
//!        │   + short_term_attn(key, last_key, last_value)
//!        │   ──► segment(f16, f8, f4, readout, hidden) ──► hidden', ab
//!        │
//!        └─ si is_mem_frame :
//!              encode_value(image, f16, hidden, ab) ──► value, hidden'
//! ```
//!
//! Trois écarts au code amont, tous **volontaires et documentés** :
//!
//! * `end` (dernière frame du plan) n'est pas transmis : il ne neutralise que la
//!   mémorisation, qui est de toute façon effacée par `reset_sequence()` à la
//!   coupe suivante — la sortie du pas, elle, n'en dépend jamais ;
//! * les graphes C2 ont des **formes statiques** : la frame est ramenée à la
//!   résolution de travail dérivée de son rapport d'aspect
//!   ([`crate::input::work_size`]), qui remplit la forme complétée des graphes ;
//! * `is_deep_update` est toujours vrai dans notre configuration
//!   (`deep_update_every < 0`), c'est donc la variante exportée.

use colorvid_ai_core::error::AiCoreError;
use colorvid_ai_core::model::{
    ColorizationInput, ColorizationOutput, EngineProfile, ImageColorizer,
};
use colorvid_ai_core::provider::{build_session, ProviderChoice};
use ort::session::Session;
use ort::value::{Tensor, ValueType};

use crate::input::{self, PreparedStep};
use crate::state::{ColorMNetState, EngineConfig, MemoryFrame};
use crate::QualityError;

/// Les deux canaux de chroma propagés comme deux « objets » (étiquettes 1-based
/// de l'amont : `set_all_labels(range(1, 3))`).
pub const CHROMA_OBJECTS: [usize; 2] = [1, 2];

/// Versions des sorties attendues des quatre graphes (noms `out0…`, C2).
const KEY_OUTPUTS: [&str; 6] = ["out0", "out1", "out2", "out3", "out4", "out5"];
const ATTN_OUTPUT: &str = "out0";
const SEGMENT_HIDDEN: &str = "out0";
const SEGMENT_AB: &str = "out2";
const VALUE_OUTPUT: &str = "out0";
const VALUE_HIDDEN: &str = "out1";

/// Formes lues sur les graphes au chargement (aucune constante devinée).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shapes {
    pub padded_width: usize,
    pub padded_height: usize,
    /// Canaux de clé (`C^k`) et de valeur (`C^v`).
    pub ck: usize,
    pub cv: usize,
    /// Canaux de la mémoire vive (`C^h`).
    pub ch: usize,
    /// Résolution de la carte de clés (`H/P`, `W/P`).
    pub key_height: usize,
    pub key_width: usize,
    /// Nombre d'objets propagés.
    pub objects: usize,
    /// Forme de `f8` et `f4` (sorties 4 et 5 de `encode_key`).
    pub f8: [usize; 3],
    pub f4: [usize; 3],
}

impl Shapes {
    /// `H*W` de la carte de clés.
    pub fn key_hw(&self) -> usize {
        self.key_height * self.key_width
    }

    /// Taille de la mémoire vive : `num_objects × C^h × H/P × W/P`.
    pub fn hidden_len(&self) -> usize {
        self.objects * self.ch * self.key_hw()
    }

    pub fn value_len(&self) -> usize {
        self.objects * self.cv * self.key_hw()
    }
}

/// Trace du dernier pas — **diagnostic uniquement**, remplie seulement quand la
/// variable d'environnement `COLORVID_QUALITY_TRACE=1` est positionnée au
/// chargement (sinon coût nul). Sert à comparer pas à pas avec la référence
/// PyTorch quand un écart apparaît (cf. `tests/orchestration_matches_pytorch.rs`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StepTrace {
    pub step: usize,
    /// `match_memory` seul (mémoire clé/valeur).
    pub memory_readout: Vec<f32>,
    /// `short_term_attn` seul, remis en `[objets, cv, hw]`.
    pub short_term: Vec<f32>,
    /// Mémoire vive **en entrée** de `segment`.
    pub hidden_in: Vec<f32>,
    /// Mémoire vive produite par `segment`.
    pub hidden_out: Vec<f32>,
    /// Sortie du pas (`tanh`, ±1), à la résolution complétée.
    pub ab: Vec<f32>,
    /// `value` produite par `encode_value` (pas de mémorisation seulement).
    pub value: Vec<f32>,
}

/// Répartition du temps par étage — **diagnostic uniquement**, rempli seulement si
/// `COLORVID_QUALITY_PROFILE=1` au chargement (sinon aucun coût : les `Instant` ne
/// sont même pas pris).
///
/// Sert à savoir *où* passe le temps par frame : les quatre graphes, le readout
/// mémoire, ou la glue Rust. Sans cette mesure, on optimise à l'aveugle.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct StageTimings {
    pub prepare_ns: u64,
    pub encode_key_ns: u64,
    pub match_memory_ns: u64,
    pub short_term_ns: u64,
    pub segment_ns: u64,
    pub encode_value_ns: u64,
    pub commit_ns: u64,
    pub output_ns: u64,
    pub steps: u64,
}

impl StageTimings {
    /// Total mesuré, en secondes — sert à vérifier que la somme des étages
    /// explique bien le temps de rendu annoncé.
    pub fn total_secs(&self) -> f64 {
        (self.prepare_ns
            + self.encode_key_ns
            + self.match_memory_ns
            + self.short_term_ns
            + self.segment_ns
            + self.encode_value_ns
            + self.commit_ns
            + self.output_ns) as f64
            / 1e9
    }

    /// Une ligne par étage, du plus coûteux au moins coûteux.
    pub fn report(&self) -> String {
        let steps = self.steps.max(1) as f64;
        let mut rows = vec![
            ("préparation (Lab + complétion)", self.prepare_ns),
            ("encode_key (graphe)", self.encode_key_ns),
            ("readout mémoire (Rust)", self.match_memory_ns),
            ("short_term_attn (graphe)", self.short_term_ns),
            ("segment (graphe)", self.segment_ns),
            ("encode_value (graphe)", self.encode_value_ns),
            ("commit mémoire (Rust)", self.commit_ns),
            ("sortie (recadrage + RGB)", self.output_ns),
        ];
        rows.sort_by_key(|row| std::cmp::Reverse(row.1));
        let total: u64 = rows.iter().map(|r| r.1).sum();
        let mut out = format!(
            "profil moteur — {} pas, {:.1} s mesurés ({:.3} s/pas)\n",
            self.steps,
            total as f64 / 1e9,
            total as f64 / 1e9 / steps
        );
        for (name, ns) in rows {
            let share = if total == 0 { 0.0 } else { ns as f64 * 100.0 / total as f64 };
            out.push_str(&format!(
                "  {name:34} {:8.1} ms/pas  {:5.1} %\n",
                ns as f64 / 1e6 / steps,
                share
            ));
        }
        out
    }
}

/// Moteur « quality » : quatre sessions ONNX sans état + l'état récurrent en Rust.
pub struct ColorMNetEngine {
    config: EngineConfig,
    state: ColorMNetState,
    shapes: Shapes,
    trace: Option<StepTrace>,
    profile: Option<StageTimings>,
    encode_key: Session,
    short_term_attn: Session,
    segment: Session,
    encode_value: Session,
}

/// Sorties de `encode_key`.
struct KeyOutput {
    key: Vec<f32>,
    shrinkage: Vec<f32>,
    selection: Vec<f32>,
    f16: Vec<f32>,
    f8: Vec<f32>,
    f4: Vec<f32>,
}

/// Sorties de `segment`.
struct SegmentOutput {
    hidden: Vec<f32>,
    ab: Vec<f32>,
}

/// Sorties de `encode_value`.
struct ValueOutput {
    value: Vec<f32>,
    hidden: Vec<f32>,
}

fn dims(ty: &ValueType) -> Vec<i64> {
    match ty {
        ValueType::Tensor { shape, .. } => shape.iter().copied().collect(),
        _ => Vec::new(),
    }
}

fn input_shape(session: &Session, name: &str) -> Result<Vec<i64>, AiCoreError> {
    session
        .inputs()
        .iter()
        .find(|i| i.name() == name)
        .map(|i| dims(i.dtype()))
        .ok_or_else(|| AiCoreError::ModelLoad(format!("entrée `{name}` absente du graphe")))
}

fn output_shape(session: &Session, name: &str) -> Result<Vec<i64>, AiCoreError> {
    session
        .outputs()
        .iter()
        .find(|o| o.name() == name)
        .map(|o| dims(o.dtype()))
        .ok_or_else(|| AiCoreError::ModelLoad(format!("sortie `{name}` absente du graphe")))
}

fn positive(shape: &[i64], what: &str) -> Result<Vec<usize>, AiCoreError> {
    shape
        .iter()
        .map(|d| {
            usize::try_from(*d).map_err(|_| {
                AiCoreError::Shape(format!("{what} : dimension dynamique ou négative ({d})"))
            })
        })
        .collect()
}

impl ColorMNetEngine {
    /// Charge les quatre graphes (checksums vérifiés en amont par
    /// [`crate::GraphSet::verify`]) et lit leurs formes.
    ///
    /// `intra_threads` borne le parallélisme : le moteur quality est gourmand en
    /// mémoire, un rendu concurrent doit rester possible.
    pub fn load(
        config: EngineConfig,
        graphs: &crate::GraphSet,
        provider: ProviderChoice,
        intra_threads: Option<usize>,
    ) -> Result<Self, AiCoreError> {
        let mut sessions = Vec::with_capacity(crate::GRAPH_FILES.len());
        for file in crate::GRAPH_FILES {
            let path = graphs.graph_path(file);
            tracing::info!(graph = %file, "chargement d'un graphe colorvid quality");
            sessions.push(build_session(&path, provider, intra_threads)?);
        }
        let [encode_key, short_term_attn, segment, encode_value] =
            <[Session; 4]>::try_from(sessions).map_err(|_| {
                AiCoreError::ModelLoad("jeu de graphes quality incomplet".to_string())
            })?;
        let shapes = Self::read_shapes(&encode_key, &short_term_attn, &segment, &encode_value)?;
        let mut state = ColorMNetState::new();
        state.reset_with(&config);
        let trace = std::env::var("COLORVID_QUALITY_TRACE")
            .map(|v| v == "1")
            .unwrap_or(false)
            .then(StepTrace::default);
        let profile = std::env::var("COLORVID_QUALITY_PROFILE")
            .map(|v| v == "1")
            .unwrap_or(false)
            .then(StageTimings::default);
        Ok(Self {
            config,
            state,
            shapes,
            trace,
            profile,
            encode_key,
            short_term_attn,
            segment,
            encode_value,
        })
    }

    /// Lit et **valide** les formes : toute incohérence entre les graphes est
    /// signalée au chargement, pas au milieu d'un rendu.
    fn read_shapes(
        encode_key: &Session,
        short_term_attn: &Session,
        segment: &Session,
        encode_value: &Session,
    ) -> Result<Shapes, AiCoreError> {
        let image = positive(&input_shape(encode_key, "image")?, "encode_key.image")?;
        if image.len() != 4 {
            return Err(AiCoreError::Shape(format!(
                "encode_key.image : rang {} au lieu de 4",
                image.len()
            )));
        }
        let key = positive(&output_shape(encode_key, KEY_OUTPUTS[0])?, "encode_key.out0")?;
        let value = positive(&output_shape(encode_value, VALUE_OUTPUT)?, "encode_value.out0")?;
        let hidden = positive(&output_shape(segment, SEGMENT_HIDDEN)?, "segment.out0")?;
        let ab = positive(&output_shape(segment, SEGMENT_AB)?, "segment.out2")?;
        let q = positive(&input_shape(short_term_attn, "q")?, "short_term_attn.q")?;
        let v = positive(&input_shape(short_term_attn, "v")?, "short_term_attn.v")?;
        let f8 = positive(&output_shape(encode_key, KEY_OUTPUTS[4])?, "encode_key.out4")?;
        let f4 = positive(&output_shape(encode_key, KEY_OUTPUTS[5])?, "encode_key.out5")?;
        // Toutes les sorties utilisées doivent exister : `outputs[name]` panique
        // sinon, et une panique au milieu d'un rendu serait pire qu'une erreur.
        for name in KEY_OUTPUTS {
            output_shape(encode_key, name)?;
        }
        output_shape(encode_value, VALUE_HIDDEN)?;

        let shapes = Shapes {
            padded_height: image[2],
            padded_width: image[3],
            ck: key[1],
            key_height: key[2],
            key_width: key[3],
            cv: value[2],
            ch: hidden[2],
            objects: hidden[1],
            f8: [f8[1], f8[2], f8[3]],
            f4: [f4[1], f4[2], f4[3]],
        };
        let expect = |what: &str, got: &[usize], want: &[usize]| -> Result<(), AiCoreError> {
            if got != want {
                return Err(AiCoreError::Shape(format!(
                    "{what} : {got:?} au lieu de {want:?} (graphes incohérents)"
                )));
            }
            Ok(())
        };
        let shrinkage = positive(&output_shape(encode_key, KEY_OUTPUTS[1])?, "encode_key.out1")?;
        expect("encode_key.out1 (shrinkage)", &shrinkage[2..], &[shapes.key_height, shapes.key_width])?;
        expect("encode_key.out0 (clé)", &key[1..], &[shapes.ck, shapes.key_height, shapes.key_width])?;
        expect(
            "encode_value.out0",
            &value[2..],
            &[shapes.cv, shapes.key_height, shapes.key_width],
        )?;
        expect(
            "segment.out2",
            &ab[2..],
            &[shapes.padded_height, shapes.padded_width],
        )?;
        expect(
            "short_term_attn.q",
            &q[1..],
            &[shapes.ck, shapes.key_height, shapes.key_width],
        )?;
        expect(
            "short_term_attn.v",
            &v[1..],
            &[shapes.objects * shapes.cv, shapes.key_height, shapes.key_width],
        )?;
        Ok(shapes)
    }

    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    pub fn state(&self) -> &ColorMNetState {
        &self.state
    }

    pub fn shapes(&self) -> &Shapes {
        &self.shapes
    }

    /// Trace du dernier pas, si `COLORVID_QUALITY_TRACE=1`.
    /// Ajoute la durée d'un étage. `start` vaut `None` quand le profilage est
    /// éteint — aucun `Instant` n'est alors pris, donc aucun coût.
    fn mark(
        profile: &mut Option<StageTimings>,
        start: Option<std::time::Instant>,
        slot: fn(&mut StageTimings) -> &mut u64,
    ) {
        if let (Some(profile), Some(start)) = (profile.as_mut(), start) {
            *slot(profile) += start.elapsed().as_nanos() as u64;
        }
    }

    fn ticking(&self) -> Option<std::time::Instant> {
        self.profile.is_some().then(std::time::Instant::now)
    }

    pub fn timings(&self) -> Option<&StageTimings> {
        self.profile.as_ref()
    }

    pub fn last_trace(&self) -> Option<&StepTrace> {
        self.trace.as_ref()
    }

    fn inference<T>(result: Result<T, ort::Error>) -> Result<T, AiCoreError> {
        result.map_err(|e| AiCoreError::Inference(e.to_string()))
    }

    fn run_encode_key(&mut self, image: &[f32]) -> Result<KeyOutput, AiCoreError> {
        let (w, h) = (self.shapes.padded_width, self.shapes.padded_height);
        let tensor = Self::inference(Tensor::from_array(([1usize, 3, h, w], image.to_vec())))?;
        let outputs = Self::inference(self.encode_key.run(ort::inputs!["image" => tensor]))?;
        let take = |name: &str| -> Result<Vec<f32>, AiCoreError> {
            let (_, data) = Self::inference(outputs[name].try_extract_tensor::<f32>())?;
            Ok(data.to_vec())
        };
        Ok(KeyOutput {
            key: take(KEY_OUTPUTS[0])?,
            shrinkage: take(KEY_OUTPUTS[1])?,
            selection: take(KEY_OUTPUTS[2])?,
            f16: take(KEY_OUTPUTS[3])?,
            f8: take(KEY_OUTPUTS[4])?,
            f4: take(KEY_OUTPUTS[5])?,
        })
    }

    /// `short_term_attn(key, last_key, last_value)` remis en `[objects, cv, hw]`.
    ///
    /// La sortie du graphe est `[hw, 1, objects*cv]` ; l'amont fait
    /// `permute(1, 2, 0).view(1, objects, cv, h, w)`.
    fn run_short_term(
        &mut self,
        query: &[f32],
        last_key: &[f32],
        last_value: &[f32],
    ) -> Result<Vec<f32>, AiCoreError> {
        let (kh, kw) = (self.shapes.key_height, self.shapes.key_width);
        let (ck, cv, no) = (self.shapes.ck, self.shapes.cv, self.shapes.objects);
        let q = Self::inference(Tensor::from_array(([1usize, ck, kh, kw], query.to_vec())))?;
        let k = Self::inference(Tensor::from_array(([1usize, ck, kh, kw], last_key.to_vec())))?;
        let v = Self::inference(Tensor::from_array((
            [1usize, no * cv, kh, kw],
            last_value.to_vec(),
        )))?;
        let outputs =
            Self::inference(self.short_term_attn.run(ort::inputs!["q" => q, "k" => k, "v" => v]))?;
        let (shape, data) =
            Self::inference(outputs[ATTN_OUTPUT].try_extract_tensor::<f32>())?;
        let hw = kh * kw;
        if data.len() != hw * no * cv {
            return Err(AiCoreError::Shape(format!(
                "short_term_attn : {} valeurs au lieu de {} (forme {:?})",
                data.len(),
                hw * no * cv,
                shape
            )));
        }
        let mut out = vec![0.0f32; data.len()];
        for j in 0..hw {
            for o in 0..no {
                for c in 0..cv {
                    out[o * cv * hw + c * hw + j] = data[j * no * cv + o * cv + c];
                }
            }
        }
        Ok(out)
    }

    fn run_segment(
        &mut self,
        f16: &[f32],
        f8: &[f32],
        f4: &[f32],
        memory_readout: &[f32],
        hidden: &[f32],
    ) -> Result<SegmentOutput, AiCoreError> {
        let (ph, pw) = (self.shapes.padded_height, self.shapes.padded_width);
        let (kh, kw) = (self.shapes.key_height, self.shapes.key_width);
        let (no, cv, ch) = (self.shapes.objects, self.shapes.cv, self.shapes.ch);
        let f16_t = Self::inference(Tensor::from_array((
            [1usize, f16.len() / (kh * kw), kh, kw],
            f16.to_vec(),
        )))?;
        let f8_t = Self::inference(Tensor::from_array((
            [1usize, self.shapes.f8[0], self.shapes.f8[1], self.shapes.f8[2]],
            f8.to_vec(),
        )))?;
        let f4_t = Self::inference(Tensor::from_array((
            [1usize, self.shapes.f4[0], self.shapes.f4[1], self.shapes.f4[2]],
            f4.to_vec(),
        )))?;
        let mem_t = Self::inference(Tensor::from_array((
            [1usize, no, cv, kh, kw],
            memory_readout.to_vec(),
        )))?;
        let hidden_t =
            Self::inference(Tensor::from_array(([1usize, no, ch, kh, kw], hidden.to_vec())))?;
        let outputs = Self::inference(self.segment.run(ort::inputs![
            "f16" => f16_t,
            "f8" => f8_t,
            "f4" => f4_t,
            "memory_readout" => mem_t,
            "hidden" => hidden_t,
        ]))?;
        let (_, h) = Self::inference(outputs[SEGMENT_HIDDEN].try_extract_tensor::<f32>())?;
        let (_, ab) = Self::inference(outputs[SEGMENT_AB].try_extract_tensor::<f32>())?;
        if ab.len() != no * ph * pw {
            return Err(AiCoreError::Shape(format!(
                "segment : {} valeurs ab au lieu de {}",
                ab.len(),
                no * ph * pw
            )));
        }
        Ok(SegmentOutput {
            hidden: h.to_vec(),
            ab: ab.to_vec(),
        })
    }

    fn run_encode_value(
        &mut self,
        image: &[f32],
        f16: &[f32],
        hidden: &[f32],
        mask: &[f32],
    ) -> Result<ValueOutput, AiCoreError> {
        let (ph, pw) = (self.shapes.padded_height, self.shapes.padded_width);
        let (kh, kw) = (self.shapes.key_height, self.shapes.key_width);
        let (no, cv, ch) = (self.shapes.objects, self.shapes.cv, self.shapes.ch);
        let image_t =
            Self::inference(Tensor::from_array(([1usize, 3, ph, pw], image.to_vec())))?;
        let f16_t = Self::inference(Tensor::from_array((
            [1usize, f16.len() / (kh * kw), kh, kw],
            f16.to_vec(),
        )))?;
        let hidden_t =
            Self::inference(Tensor::from_array(([1usize, no, ch, kh, kw], hidden.to_vec())))?;
        let mask_t =
            Self::inference(Tensor::from_array(([1usize, no, ph, pw], mask.to_vec())))?;
        let outputs = Self::inference(self.encode_value.run(ort::inputs![
            "image" => image_t,
            "f16" => f16_t,
            "hidden" => hidden_t,
            "mask" => mask_t,
        ]))?;
        let (_, value) = Self::inference(outputs[VALUE_OUTPUT].try_extract_tensor::<f32>())?;
        let (_, h) = Self::inference(outputs[VALUE_HIDDEN].try_extract_tensor::<f32>())?;
        if value.len() != no * cv * kh * kw {
            return Err(AiCoreError::Shape(format!(
                "encode_value : {} valeurs au lieu de {}",
                value.len(),
                no * cv * kh * kw
            )));
        }
        Ok(ValueOutput {
            value: value.to_vec(),
            hidden: h.to_vec(),
        })
    }
}

/// Convertit une erreur interne en erreur d'inférence `ai-core`.
fn quality<T, E: Into<QualityError>>(result: Result<T, E>) -> Result<T, AiCoreError> {
    result.map_err(|e| AiCoreError::Inference(e.into().to_string()))
}

impl ImageColorizer for ColorMNetEngine {
    /// Un pas d'inférence complet (une frame). L'état récurrent est conservé
    /// d'un appel à l'autre ; `reset_sequence()` marque le début d'un plan.
    fn colorize_image(
        &mut self,
        input: &ColorizationInput,
    ) -> Result<ColorizationOutput, AiCoreError> {
        let step = self.state.step();
        let has_exemplar = step == 0;
        if has_exemplar && input.reference.is_none() {
            return quality(Err(QualityError::MissingExemplar));
        }
        let tick = self.ticking();
        let prepared: PreparedStep = input::prepare(
            &input.grayscale_rgb,
            input.reference.as_ref(),
            (self.shapes.padded_width, self.shapes.padded_height),
        );
        // `input::prepare` dérive la résolution de travail **pour que** la
        // complétion retombe sur la forme des graphes : ce contrôle est la
        // garantie que ce contrat n'a pas dérivé.
        if prepared.padded() != (self.shapes.padded_width, self.shapes.padded_height) {
            return Err(AiCoreError::Shape(format!(
                "résolution de travail {}×{} complétée en {}×{}, attendu {}×{}",
                prepared.width,
                prepared.height,
                prepared.padded().0,
                prepared.padded().1,
                self.shapes.padded_width,
                self.shapes.padded_height
            )));
        }

        Self::mark(&mut self.profile, tick, |p| &mut p.prepare_ns);

        let phase = self.state.phase(&self.config, has_exemplar, false);
        let tick = self.ticking();
        let key_out = self.run_encode_key(&prepared.image)?;
        Self::mark(&mut self.profile, tick, |p| &mut p.encode_key_ns);
        if let Some(trace) = self.trace.as_mut() {
            *trace = StepTrace {
                step,
                ..StepTrace::default()
            };
        }

        let mut ab: Vec<f32> = Vec::new();
        if phase.needs_segment {
            let tick = self.ticking();
            let readout = quality(
                self.state
                    .memory()
                    .match_memory(&key_out.key, Some(&key_out.selection), self.config.top_k)
                    .map_err(QualityError::from),
            )?;
            Self::mark(&mut self.profile, tick, |p| &mut p.match_memory_ns);
            let last_key = self
                .state
                .last_key()
                .ok_or_else(|| AiCoreError::Inference("mémoire court terme vide".into()))?
                .to_vec();
            let last_value = self
                .state
                .last_value()
                .ok_or_else(|| AiCoreError::Inference("mémoire court terme vide".into()))?
                .to_vec();
            let tick = self.ticking();
            let short = self.run_short_term(&key_out.key, &last_key, &last_value)?;
            Self::mark(&mut self.profile, tick, |p| &mut p.short_term_ns);
            if let Some(trace) = self.trace.as_mut() {
                trace.memory_readout = readout.data.clone();
                trace.short_term = short.clone();
            }
            let memory_readout: Vec<f32> = readout
                .data
                .iter()
                .zip(&short)
                .map(|(m, s)| m + s)
                .collect();
            let hidden = self
                .state
                .hidden()
                .ok_or_else(|| AiCoreError::Inference("mémoire vive absente".into()))?
                .to_vec();
            let tick = self.ticking();
            let segmented =
                self.run_segment(&key_out.f16, &key_out.f8, &key_out.f4, &memory_readout, &hidden)?;
            Self::mark(&mut self.profile, tick, |p| &mut p.segment_ns);
            if let Some(trace) = self.trace.as_mut() {
                trace.hidden_in = hidden.clone();
                trace.hidden_out = segmented.hidden.clone();
            }
            if phase.is_normal_update {
                self.state.set_hidden(segmented.hidden);
            }
            ab = segmented.ab;
        }

        if has_exemplar {
            // L'amont : `create_hidden_state(2, key)` (zéros) puis la sortie de la
            // frame 0 est l'exemplaire lui-même.
            self.state.set_hidden(vec![0.0; self.shapes.hidden_len()]);
            ab = prepared
                .exemplar
                .clone()
                .ok_or_else(|| AiCoreError::Inference("exemplaire absent".into()))?;
        }

        let mut frame: Option<MemoryFrame> = None;
        if phase.is_mem_frame {
            let hidden = self
                .state
                .hidden()
                .ok_or_else(|| AiCoreError::Inference("mémoire vive absente à la mémorisation".into()))?
                .to_vec();
            let tick = self.ticking();
            let valued = self.run_encode_value(&prepared.image, &key_out.f16, &hidden, &ab)?;
            Self::mark(&mut self.profile, tick, |p| &mut p.encode_value_ns);
            if let Some(trace) = self.trace.as_mut() {
                trace.value = valued.value.clone();
            }
            frame = Some(MemoryFrame {
                objects: CHROMA_OBJECTS.to_vec(),
                key: key_out.key,
                shrinkage: key_out.shrinkage,
                value: valued.value,
            });
            if phase.is_deep_update {
                self.state.set_hidden(valued.hidden);
            }
        }
        // Le calendrier avance à **chaque** frame, mémorisée ou non.
        let tick = self.ticking();
        quality(self.state.commit(&self.config, phase, frame.as_ref()))?;
        Self::mark(&mut self.profile, tick, |p| &mut p.commit_ns);

        // `ab` est à la résolution **complétée** (c'est ce que produisent
        // `segment` et l'exemplaire) ; le recadrage vient ensuite.
        let (pw, ph) = prepared.padded();
        if ab.len() != 2 * pw * ph {
            return Err(AiCoreError::Shape(format!(
                "sortie ab : {} valeurs au lieu de {}",
                ab.len(),
                2 * pw * ph
            )));
        }
        if let Some(trace) = self.trace.as_mut() {
            trace.ab = ab.clone();
        }
        let tick = self.ticking();
        let cropped = input::unpad_chw(&ab, 2, prepared.width, prepared.height, prepared.pad);
        // Le modèle prédit `ab` dans ±1 (`tanh`) : retour aux unités CIELAB.
        let values: Vec<f32> = cropped.iter().map(|v| v * 110.0).collect();
        Self::mark(&mut self.profile, tick, |p| &mut p.output_ns);
        if let Some(profile) = self.profile.as_mut() {
            profile.steps += 1;
        }
        Ok(ColorizationOutput::LabAb {
            values,
            width: prepared.width as u32,
            height: prepared.height as u32,
        })
    }

    /// ColorMNet est un moteur **à exemplaire** : la référence n'est pas
    /// optionnelle, elle porte la couleur du plan.
    fn uses_reference(&self) -> bool {
        true
    }

    /// Ni nettoyage spatial, ni stabilisation par flux, ni propagation de
    /// références : le modèle produit lui-même sa cohérence et prend
    /// l'exemplaire en entrée (plan §3.3). Reste le **gain de chroma borné vers
    /// la chroma de l'exemplaire**, recette retenue au C1bis (butée 1,8,
    /// lissage 0,10) — les étages DDColor sont calibrés 1,12 / 0,12.
    /// Et la mémoire du modèle impose une réinitialisation à chaque coupe.
    fn pipeline_profile(&self) -> EngineProfile {
        EngineProfile {
            spatial_cleanup: false,
            temporal_stabilizer: false,
            reference_guides: false,
            chroma_gain: Some((1.8, 0.10)),
            shot_resets: true,
        }
    }

    fn reset_sequence(&mut self) {
        self.state.reset_with(&self.config);
        tracing::debug!("état quality réinitialisé (changement de plan)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::StoreError;

    #[test]
    fn shapes_derive_the_state_dimensions() {
        let shapes = Shapes {
            padded_width: 672,
            padded_height: 448,
            ck: 64,
            cv: 512,
            ch: 64,
            key_height: 28,
            key_width: 42,
            objects: 2,
            f8: [512, 56, 84],
            f4: [256, 112, 168],
        };
        assert_eq!(shapes.key_hw(), 1176);
        assert_eq!(shapes.hidden_len(), 2 * 64 * 1176);
        assert_eq!(shapes.value_len(), 2 * 512 * 1176);
    }

    #[test]
    fn missing_dynamic_dimensions_are_reported() {
        let err = positive(&[1, 3, -1, 672], "test").unwrap_err();
        assert!(matches!(err, AiCoreError::Shape(_)), "{err}");
        assert_eq!(positive(&[1, 3, 448, 672], "test").unwrap(), vec![1, 3, 448, 672]);
    }

    #[test]
    fn store_errors_become_quality_errors() {
        let err =
            quality::<(), QualityError>(Err(QualityError::State(StoreError::Empty))).unwrap_err();
        assert!(err.to_string().contains("mémoire"), "{err}");
    }
}
