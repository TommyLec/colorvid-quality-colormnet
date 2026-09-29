//! Validation **pas à pas** de l'orchestration contre l'exécution PyTorch de
//! référence.
//!
//! Ce test est volontairement `#[ignore]` : il a besoin des graphes ONNX
//! exportés (`.j0/colormnet/onnx`, ~510 Mo) et des références produites par
//! `InferenceCore.step` (`.j0/colormnet/c4_ref`), qui ne sont pas versionnés.
//!
//! Il vérifie **deux choses distinctes** :
//!
//! 1. la préparation d'entrée Rust (Lab `skimage`, normalisation, complétion
//!    112) est identique au tenseur réellement vu par le réseau en Python ;
//! 2. la sortie `ab` de chaque pas est identique à celle de l'amont — donc
//!    l'enchaînement `encode_key` → mémoire + attention court terme →
//!    `segment` → `encode_value`, l'état récurrent et l'injection de
//!    l'exemplaire sont corrects.
//!
//! Les écarts mesurés (CPU, fp32) sont de l'ordre de 10⁻⁵ : la tolérance
//! retenue (10⁻³) laisse la marge nécessaire à un autre provider d'exécution.
//!
//! La sortie détaille, pas par pas, l'écart sur `ab` **et** sur chaque tenseur
//! intermédiaire (lecture mémoire, attention court terme, mémoire vive, valeur).
//!
//! ```bash
//! CARGO_HOME=$PWD/.j0/cargo-home \
//!   cargo test -p colorvid-quality-colormnet --test orchestration_matches_pytorch \
//!   -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use colorvid_ai_core::model::{ColorizationInput, ColorizationOutput, ImageColorizer};
use colorvid_ai_core::provider::ProviderChoice;
use colorvid_types::config::GpuProvider;

/// EP du test : `cpu` par défaut, `migraphx` via `COLORVID_TEST_PROVIDER`.
fn provider_under_test() -> ProviderChoice {
    match std::env::var("COLORVID_TEST_PROVIDER").as_deref() {
        Ok("migraphx") => ProviderChoice::Gpu(GpuProvider::Migraphx),
        Ok("auto") => ProviderChoice::Gpu(GpuProvider::Auto),
        _ => ProviderChoice::Cpu,
    }
}
use colorvid_quality_colormnet::input;
use colorvid_quality_colormnet::state::EngineConfig;
use colorvid_quality_colormnet::{ColorMNetEngine, GraphSet};

/// Tolérance sur `ab` (±1) et sur les tenseurs intermédiaires.
const TOLERANCE: f32 = 1e-3;
/// Tolérance sur la préparation d'entrée : du bruit d'arrondi f32, rien d'autre.
const TOLERANCE_ENTREE: f32 = 1e-5;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("racine du workspace")
        .to_path_buf()
}

fn override_or(var: &str, default: PathBuf) -> PathBuf {
    std::env::var(var).map(PathBuf::from).unwrap_or(default)
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("lecture de {} impossible : {e}", path.display()));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "longueurs différentes : {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn path_of(dir: &Path, stem: &str) -> Option<PathBuf> {
    let path = dir.join(format!("{stem}.bin"));
    path.exists().then_some(path)
}

fn load_png(path: &Path) -> image::RgbImage {
    image::open(path)
        .unwrap_or_else(|e| panic!("ouverture de {} impossible : {e}", path.display()))
        .to_rgb8()
}

#[test]
#[ignore = "requiert les graphes ONNX (.j0/colormnet/onnx) et les références PyTorch (.j0/colormnet/c4_ref)"]
fn orchestration_reproduces_the_pytorch_step_outputs() {
    let root = workspace_root();
    let reference_dir = override_or("COLORVID_QUALITY_REF", root.join(".j0/colormnet/c4_ref"));
    let graphs_dir = override_or("COLORVID_QUALITY_GRAPHS", root.join(".j0/colormnet/onnx"));
    if !reference_dir.join("meta.json").exists() || !graphs_dir.join("encode_key.onnx").exists() {
        eprintln!(
            "graphes ou références absents ({} / {}) — rien à vérifier",
            graphs_dir.display(),
            reference_dir.display()
        );
        return;
    }

    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(reference_dir.join("meta.json")).unwrap())
            .expect("meta.json");
    let steps = meta["frames"].as_u64().expect("frames") as usize;

    // La trace pas à pas rend la sortie du test exploitable en cas d'écart.
    std::env::set_var("COLORVID_QUALITY_TRACE", "1");
    let mut engine = ColorMNetEngine::load(
        EngineConfig::default(),
        &GraphSet::new(&graphs_dir),
        // `COLORVID_TEST_PROVIDER=migraphx` : rejoue la même référence PyTorch sur
        // le GPU. C'est le **test différentiel** du moteur : chaque étage est
        // comparé séparément (`lecture`, `court`, `memoire`, `ab`), donc un écart
        // de calcul GPU est nommé et chiffré au lieu d'être noyé dans la sortie.
        provider_under_test(),
        None,
    )
    .expect("chargement des quatre graphes");
    let exemplar = load_png(&reference_dir.join("exemplaire.png"));

    // --- 1. préparation d'entrée, comparée au tenseur vu par PyTorch --------
    let first = load_png(&reference_dir.join("frames/00000.png"));
    let prepared = input::prepare(&first, Some(&exemplar), (672, 448));
    let image_diff = max_abs_diff(&prepared.image, &read_f32(&reference_dir.join("image0.bin")));
    let exemplar_diff = max_abs_diff(
        prepared.exemplar.as_ref().expect("exemplaire préparé"),
        &read_f32(&reference_dir.join("exemplaire.bin")),
    );
    eprintln!("préparation : écart L {image_diff:.2e} | écart exemplaire {exemplar_diff:.2e}");
    assert!(image_diff < TOLERANCE_ENTREE, "canal L divergent : {image_diff}");
    assert!(
        exemplar_diff < TOLERANCE_ENTREE,
        "exemplaire divergent : {exemplar_diff}"
    );

    // --- 2. le pas d'inférence complet -------------------------------------
    let mut worst_ab = 0.0f32;
    let mut worst_lecture = 0.0f32;
    let mut worst_court = 0.0f32;
    for step in 0..steps {
        let frame = load_png(&reference_dir.join(format!("frames/{step:05}.png")));
        let output = engine
            .colorize_image(&ColorizationInput {
                grayscale_rgb: frame,
                reference: Some(exemplar.clone()),
            })
            .unwrap_or_else(|e| panic!("pas {step} : {e}"));
        let ColorizationOutput::LabAb { values, .. } = output else {
            panic!("sortie inattendue au pas {step}");
        };
        // le moteur renvoie des unités Lab, la référence est en ±1 (`tanh`)
        let mine: Vec<f32> = values.iter().map(|v| v / 110.0).collect();
        let expected = read_f32(&reference_dir.join(format!("ab{step}.bin")));
        let diff = max_abs_diff(&mine, &expected);
        worst_ab = worst_ab.max(diff);

        // Diagnostic pas à pas : où l'écart apparaît-il, s'il y en a un ?
        let mut detail = String::new();
        if let Some(trace) = engine.last_trace() {
            if let Some(path) = path_of(&reference_dir, &format!("lecture{step}")) {
                let d = max_abs_diff(&trace.memory_readout, &read_f32(&path));
                worst_lecture = worst_lecture.max(d);
                detail += &format!(" | lecture {d:.2e}");
            }
            if let Some(path) = path_of(&reference_dir, &format!("court{step}")) {
                // référence : [hw, 1, objets*cv] → [objets, cv, hw] (même
                // transposition que `run_short_term`)
                let reference_short = read_f32(&path);
                let (hw, no, cv) = (28usize * 42, 2usize, 512usize);
                let mut reordered = vec![0.0f32; reference_short.len()];
                for j in 0..hw {
                    for o in 0..no {
                        for c in 0..cv {
                            reordered[o * cv * hw + c * hw + j] =
                                reference_short[j * no * cv + o * cv + c];
                        }
                    }
                }
                let d = max_abs_diff(&trace.short_term, &reordered);
                worst_court = worst_court.max(d);
                detail += &format!(" | court {d:.2e}");
            }
            if let Some(path) = path_of(&reference_dir, &format!("memoire{step}")) {
                detail += &format!(
                    " | memoire {:.2e}",
                    max_abs_diff(&trace.hidden_out, &read_f32(&path))
                );
            }
            if let Some(path) = path_of(&reference_dir, &format!("valeur{step}")) {
                detail += &format!(" | valeur {:.2e}", max_abs_diff(&trace.value, &read_f32(&path)));
            }
        }
        eprintln!("pas {step} : écart ab max {diff:.2e}{detail}");
        assert!(diff < TOLERANCE, "pas {step} : écart ab {diff}{detail}");
    }
    eprintln!(
        "écart maximal sur {steps} pas : ab {worst_ab:.2e} | mémoire {worst_lecture:.2e} \
         | attention court terme {worst_court:.2e}"
    );
}
