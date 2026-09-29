//! A/B « quality » vs moteur par défaut, **dans le pipeline média réel**, sur
//! n'importe quelle vidéo.
//!
//! Enchaîne ce que l'application fera :
//!
//! 1. **détection des plans** (le même détecteur que l'assistant d'ancrage) ;
//! 2. **rendu DDColor** de référence, sans ancre — c'est le point de départ d'un
//!    utilisateur qui n'a rien validé ;
//! 3. **choix des ancres** : pour chaque plan, la frame dont la colorisation
//!    DDColor est au *p*-ième centile de chroma du plan (règle mesurée au §13.10 ;
//!    `--percentile`) — les vignettes d'analyse étant en niveaux de gris, la
//!    richesse colorimétrique ne peut se lire que sur un rendu couleur ;
//! 4. **rendu ColorMNet** avec ces ancres, un exemplaire par plan ;
//! 5. **extraction** des deux rendus en PNG, pour les métriques du banc.
//!
//! Les deux passes subissent le même encodage H.264 : le biais est commun.
//!
//! ```bash
//! cargo run --release -p colorvid-quality-colormnet --example quality_ab -- \
//!   --video videos/ref2.mp4 --out .j0/ref2
//! ```
//!
//! ⚠️ Outil d'ÉVALUATION : il vit dans la crate isolée et disparaît avec elle.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use colorvid_ai_core::model::{
    ColorizationInput, ColorizationOutput, Colorizer, ColorizerConfig, EngineProfile,
    ImageColorizer,
};
use colorvid_ai_core::error::AiCoreError;
use colorvid_ai_core::optical_flow::{dev_flow_paths, OpticalFlow, OpticalFlowConfig};
use colorvid_ai_core::provider::ProviderChoice;
use colorvid_ai_core::registry;
use colorvid_media::batch::{render_video, RenderInput, RenderOptions};
use colorvid_media::ffmpeg::Ffmpeg;
use colorvid_media::reference::RenderReference;
use colorvid_media::shots::{shots_from_video, Shot, ShotDetectionConfig};
use colorvid_quality_colormnet::state::{EngineConfig, MidTermPolicy};
use colorvid_quality_colormnet::{ColorMNetEngine, GraphSet};
use colorvid_types::config::{ContentCropMode, GpuProvider, TemporalStrength};

struct Args {
    video: PathBuf,
    out: PathBuf,
    anchors: Option<PathBuf>,
    graphs: PathBuf,
    only: Option<String>,
    extract_only: bool,
    /// Centile de chroma du plan où poser l'ancre (levier 1, §13.10).
    percentile: f64,
    /// Règle de choix de l'ancre : `percentile` ou `mean` (chroma moyenne du plan).
    ///
    /// `mean` est la règle **sans paramètre** : la sortie du moteur suit la chroma
    /// de son exemplaire à ~1:1 (mesuré, §13.14), donc viser la chroma moyenne du
    /// rendu DDColor du plan fait atterrir la sortie sur la référence — sans
    /// centile ajusté, donc sans dépendance à l'extrait.
    anchor_rule: String,
    /// Rendu DDColor déjà calculé, à réutiliser pour dériver les ancres.
    dd_frames: Option<PathBuf>,
    /// Expérimentation : force la stabilisation temporelle du moteur qualité
    /// (le profil livré la désactive, cf. plan §3.3).
    stabilizer: Option<TemporalStrength>,
    /// Execution provider des trois moteurs : `cpu` (défaut), `migraphx`,
    /// `webgpu` ou `auto`.
    provider: ProviderChoice,
    /// Politique de rétention mémoire du moteur quality.
    unbounded: bool,
    /// Taille de la fenêtre mémoire (pas mémorisés), si bornée.
    window: usize,
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("racine du workspace")
        .to_path_buf()
}

fn parse_args() -> Args {
    let root = workspace_root();
    let mut args = Args {
        video: root.join("videos/ref2.mp4"),
        out: root.join(".j0/qa"),
        anchors: None,
        graphs: root.join(".j0/colormnet/onnx"),
        only: None,
        extract_only: false,
        percentile: 70.0,
        anchor_rule: "mean".to_string(),
        dd_frames: None,
        stabilizer: None,
        provider: ProviderChoice::Cpu,
        unbounded: false,
        window: 10,
    };
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0;
    while index < raw.len() {
        match raw[index].as_str() {
            "--extract-only" => {
                args.extract_only = true;
                index += 1;
                continue;
            }
            "--unbounded" => {
                args.unbounded = true;
                index += 1;
                continue;
            }
            flag => {
                let value = raw
                    .get(index + 1)
                    .unwrap_or_else(|| panic!("{flag} attend une valeur"));
                match flag {
                    "--video" => args.video = PathBuf::from(value),
                    "--out" => args.out = PathBuf::from(value),
                    "--anchors" => args.anchors = Some(PathBuf::from(value)),
                    "--graphs" => args.graphs = PathBuf::from(value),
                    "--only" => args.only = Some(value.clone()),
                    "--percentile" => args.percentile = value.parse().expect("centile"),
                    "--window" => args.window = value.parse().expect("fenêtre"),
                    "--anchor-rule" => args.anchor_rule = value.clone(),
                    "--dd-frames" => args.dd_frames = Some(PathBuf::from(value)),
                    "--provider" => {
                        args.provider = match value.as_str() {
                            "cpu" => ProviderChoice::Cpu,
                            "migraphx" => ProviderChoice::Gpu(GpuProvider::Migraphx),
                            "webgpu" => ProviderChoice::Gpu(GpuProvider::WebGpu),
                            "auto" => ProviderChoice::Gpu(GpuProvider::Auto),
                            other => panic!("provider inconnu : {other}"),
                        }
                    }
                    "--stabilizer" => {
                        args.stabilizer = Some(match value.as_str() {
                            "low" => TemporalStrength::Low,
                            "standard" => TemporalStrength::Standard,
                            "strong" => TemporalStrength::Strong,
                            other => panic!("force de stabilisation inconnue : {other}"),
                        })
                    }
                    other => panic!("argument inconnu : {other}"),
                }
                index += 2;
            }
        }
    }
    args
}

/// Enveloppe d'expérimentation : mêmes graphes et même état, **profil de pipeline
/// différent**. Permet de tester un étage (ici la stabilisation temporelle) sans
/// modifier le moteur livré ni son profil.
struct ProfileOverride {
    inner: ColorMNetEngine,
    profile: EngineProfile,
}

impl ProfileOverride {
    /// Accès au chronométrage du moteur sous-jacent (profilage).
    fn timings(&self) -> Option<&colorvid_quality_colormnet::engine::StageTimings> {
        self.inner.timings()
    }
}

impl ImageColorizer for ProfileOverride {
    fn colorize_image(
        &mut self,
        input: &ColorizationInput,
    ) -> Result<ColorizationOutput, AiCoreError> {
        self.inner.colorize_image(input)
    }

    fn uses_reference(&self) -> bool {
        self.inner.uses_reference()
    }

    fn reset_sequence(&mut self) {
        self.inner.reset_sequence();
    }

    fn pipeline_profile(&self) -> EngineProfile {
        self.profile
    }
}

fn load_ddcolor(provider: ProviderChoice) -> Colorizer {
    let selection = colorvid_types::config::ModelSelection::default();
    let (model, checksum) = registry::dev_model_paths(&selection).expect("chemins du modèle");
    let descriptor = registry::descriptor(&selection.id)
        .expect("le moteur DDColor est dans le registre commun");
    let width = descriptor.capabilities.width as usize;
    Colorizer::load(&ColorizerConfig {
        model_path: model,
        checksum_path: Some(checksum),
        anchored_checksum: descriptor.expected_sha256,
        provider,
        intra_threads: Some(6),
        input_size: width,
    })
    .expect("chargement DDColor-L 512")
}

fn load_flow(provider: ProviderChoice) -> OpticalFlow {
    let (model, checksum) = dev_flow_paths();
    OpticalFlow::load(&OpticalFlowConfig {
        model_path: model,
        checksum_path: Some(checksum),
        provider,
        intra_threads: Some(4),
    })
    .expect("chargement du flux optique")
}

/// Extrait toutes les frames d'une vidéo en PNG numérotés (pour les métriques).
fn extract_all(video: &Path, dir: &Path) {
    std::fs::create_dir_all(dir).expect("dossier d'extraction");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-v",
            "error",
            "-i",
            &video.display().to_string(),
            &dir.join("f_%04d.png").display().to_string(),
        ])
        .status()
        .expect("ffmpeg");
    assert!(status.success(), "extraction de {}", video.display());
}

/// Chroma moyenne d'une frame couleur (unités Lab).
fn mean_chroma(path: &Path) -> f32 {
    let image = image::open(path).expect("frame colorisée").to_rgb8();
    let mut sum = 0.0f64;
    for pixel in image.pixels() {
        let (_, a, b) = colorvid_media::rgb_to_lab(pixel[0] as f32, pixel[1] as f32, pixel[2] as f32);
        sum += ((a * a + b * b) as f64).sqrt();
    }
    (sum / (image.width() * image.height()).max(1) as f64) as f32
}

/// Ancres dérivées du rendu DDColor : une par plan, au centile demandé.
fn build_anchors(
    shots: &[Shot],
    dd_dir: &Path,
    fps: f64,
    percentile: f64,
    rule: &str,
    out: &Path,
) -> Vec<RenderReference> {
    let frames: Vec<PathBuf> = {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dd_dir)
            .expect("rendu DDColor")
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().map(|e| e == "png").unwrap_or(false))
            .collect();
        paths.sort();
        paths
    };
    std::fs::create_dir_all(out).expect("dossier d'ancres");
    let mut references = Vec::with_capacity(shots.len());
    println!("ancres (levier 1, règle « {rule} »{}) :",
        if rule == "mean" { String::new() } else { format!(", centile {percentile:.0}") });
    for (index, shot) in shots.iter().enumerate() {
        let first = ((shot.start_secs * fps).round() as usize).min(frames.len() - 1);
        let last = ((shot.end_secs * fps).round() as usize).min(frames.len());
        let window = &frames[first..last.max(first + 1)];
        let chroma: Vec<f32> = window.iter().map(|path| mean_chroma(path)).collect();
        let target = if rule == "mean" {
            // Chroma moyenne du plan : la sortie du moteur suit sa source à ~1:1,
            // donc viser la moyenne place le plan entier au niveau de référence.
            chroma.iter().sum::<f32>() / chroma.len().max(1) as f32
        } else {
            let mut sorted = chroma.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted[((sorted.len() - 1) as f64 * percentile / 100.0).round() as usize]
        };
        let chosen = chroma
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (*a - target)
                    .abs()
                    .partial_cmp(&(*b - target).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let directory = out.join(format!("shot{:02}", index + 1));
        std::fs::create_dir_all(&directory).expect("dossier d'ancre");
        let asset = directory.join("00000.png");
        std::fs::copy(&window[chosen], &asset).expect("copie de l'ancre");
        println!(
            "  plan {} ({:.1}–{:.1} s) : frame {} — chroma {:.2} (moyenne du plan {:.2},              1re frame {:.2})",
            index + 1,
            shot.start_secs,
            shot.end_secs,
            first + chosen,
            chroma[chosen],
            target,
            chroma[0]
        );
        references.push(RenderReference {
            id: format!("shot{:02}", index + 1),
            asset_path: asset,
            anchor_secs: shot.start_secs,
            segment_start_secs: shot.start_secs,
            segment_end_secs: shot.end_secs,
            strength: 1.0,
        });
    }
    references
}

/// Références fixes, pour rejouer une mesure existante.
fn fixed_references(anchors: &Path, shots: &[Shot]) -> Vec<RenderReference> {
    shots.iter()
        .enumerate()
        .map(|(index, shot)| RenderReference {
            id: format!("shot{:02}", index + 1),
            asset_path: anchors.join(format!("shot{:02}/00000.png", index + 1)),
            anchor_secs: shot.start_secs,
            segment_start_secs: shot.start_secs,
            segment_end_secs: shot.end_secs,
            strength: 1.0,
        })
        .collect()
}

fn write_shots(shots: &[Shot], path: &Path) {
    let json: Vec<String> = shots
        .iter()
        .map(|shot| {
            format!(
                "{{\"start\": {:.3}, \"end\": {:.3}, \"anchor\": {:.3}}}",
                shot.start_secs, shot.end_secs, shot.anchor_secs
            )
        })
        .collect();
    std::fs::write(path, format!("[{}]", json.join(","))).expect("écriture shots.json");
}

fn main() {
    // Journalisation : sans abonné `tracing`, les messages d'ai-core (« chargement
    // d'un graphe »), les avertissements d'ORT et ceux de MIGraphX restent
    // invisibles — on ne sait alors pas *où* un rendu GPU échoue.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();
    // Comme l'application au démarrage : initialise ONNX Runtime une seule fois et
    // configure le cache MIGraphX. Sans cela, un rendu GPU échoue à la compilation
    // du premier graphe (chemin de cache vide).
    colorvid_ai_core::init_ort_environment();
    let args = parse_args();
    let ffmpeg = Ffmpeg::default();
    std::fs::create_dir_all(&args.out).expect("dossier de sortie");
    let probe = ffmpeg.probe(&args.video).expect("sonde de la vidéo");
    println!(
        "vidéo : {} — {}x{}, {:.2} fps, {:.1} s | provider {:?}",
        args.video.display(),
        probe.width,
        probe.height,
        probe.fps,
        probe.duration_secs,
        args.provider
    );

    // 1. Plans (même détecteur que l'assistant d'ancrage).
    let shots_dir = args.out.join("shots");
    let _ = std::fs::remove_dir_all(&shots_dir);
    std::fs::create_dir_all(&shots_dir).expect("dossier d'analyse");
    let shots = shots_from_video(&ffmpeg, &args.video, &shots_dir, &ShotDetectionConfig::default())
        .expect("détection de plans");
    println!("\n{} plan(s) détecté(s) :", shots.len());
    for (index, shot) in shots.iter().enumerate() {
        println!(
            "  plan {} : {:.2}–{:.2} s ({:.2} s)",
            index + 1,
            shot.start_secs,
            shot.end_secs,
            shot.duration_secs()
        );
    }
    write_shots(&shots, &args.out.join("shots.json"));

    if args.extract_only {
        for name in ["dd", "cm"] {
            let video = args.out.join(format!("{name}.mp4"));
            if video.exists() {
                extract_all(&video, &args.out.join(name));
                println!("frames extraites de {}", video.display());
            }
        }
        return;
    }

    // 2. Rendu DDColor de référence (sans ancre : le point de départ d'un
    //    utilisateur qui n'a rien validé), puis 3. ancres dérivées de ce rendu.
    if let Some(dd_dir) = &args.dd_frames {
        // Rendu DDColor déjà disponible : on en dérive les ancres sans relancer la
        // passe de référence.
        build_anchors(
            &shots,
            dd_dir,
            probe.fps,
            args.percentile,
            &args.anchor_rule,
            &args.out.join("anchors"),
        );
    }

    let built = if args.only.as_deref() != Some("cm") && args.dd_frames.is_none() {
        println!("\n=== DDColor-L 512 (référence) ===");
        let mut colorizer = load_ddcolor(args.provider);
        let mut flow = load_flow(args.provider);
        let cancel = AtomicBool::new(false);
        let output = args.out.join("dd.mp4");
        let last = Arc::new(AtomicU64::new(0));
        let tick = Arc::clone(&last);
        let started = std::time::Instant::now();
        let outcome = render_video(
            RenderInput {
                colorizer: &mut colorizer,
                flow: Some(&mut flow),
                ffmpeg: &ffmpeg,
                input: &args.video,
                output: &output,
                cancel: &cancel,
            },
            &RenderOptions {
                content_crop: ContentCropMode::Disabled,
                temporal_strength: TemporalStrength::Standard,
                ..RenderOptions::default()
            },
            &mut move |_, frame, total| {
                if frame / 50 > tick.load(Ordering::Relaxed) / 50 {
                    tick.store(frame, Ordering::Relaxed);
                    println!("  {frame}/{total}");
                }
            },
        )
        .expect("rendu DDColor");
        println!(
            "DDColor : {outcome:?} en {:.1} s",
            started.elapsed().as_secs_f64()
        );
        extract_all(&output, &args.out.join("dd"));
        Some(build_anchors(
            &shots,
            &args.out.join("dd"),
            probe.fps,
            args.percentile,
            &args.anchor_rule,
            &args.out.join("anchors"),
        ))
    } else {
        None
    };

    if args.only.as_deref() == Some("dd") {
        return;
    }

    // 4. Rendu ColorMNet.
    let references = match (&args.anchors, built) {
        (Some(dir), _) => fixed_references(dir, &shots),
        (None, Some(references)) => references,
        (None, None) => fixed_references(&args.out.join("anchors"), &shots),
    };

    if let Some(strength) = args.stabilizer {
        println!(
            "⚠ expérimentation : stabilisation temporelle FORCÉE ({strength:?}) — le \
             profil livré la désactive"
        );
    }
    let mid_term = if args.unbounded {
        MidTermPolicy::Unbounded
    } else {
        MidTermPolicy::Window {
            frames: args.window,
        }
    };
    println!("\n=== ColorMNet (moteur « quality »), mémoire {mid_term:?} ===");
    let config = EngineConfig {
        mid_term,
        ..EngineConfig::default()
    };
    let engine = ColorMNetEngine::load(
        config,
        &GraphSet::new(&args.graphs),
        args.provider,
        Some(8),
    )
    .expect("chargement des quatre graphes");
    // Le stabilisateur a besoin du flux optique : on ne le charge que pour le test.
    let mut stabilizer_flow = args.stabilizer.map(|_| load_flow(args.provider));
    let profile = match args.stabilizer {
        Some(_) => EngineProfile {
            temporal_stabilizer: true,
            ..engine.pipeline_profile()
        },
        None => engine.pipeline_profile(),
    };
    let mut colorizer = ProfileOverride {
        inner: engine,
        profile,
    };
    let cancel = AtomicBool::new(false);
    let output = args.out.join("cm.mp4");
    let last = Arc::new(AtomicU64::new(0));
    let tick = Arc::clone(&last);
    let started = std::time::Instant::now();
    let outcome = render_video(
        RenderInput {
            colorizer: &mut colorizer,
            flow: stabilizer_flow.as_mut(),
            ffmpeg: &ffmpeg,
            input: &args.video,
            output: &output,
            cancel: &cancel,
        },
        &RenderOptions {
            content_crop: ContentCropMode::Disabled,
            references,
            temporal_strength: args.stabilizer.unwrap_or(TemporalStrength::Standard),
            ..RenderOptions::default()
        },
        &mut move |_, frame, total| {
            if frame / 25 > tick.load(Ordering::Relaxed) / 25 {
                tick.store(frame, Ordering::Relaxed);
                println!("  {frame}/{total}");
            }
        },
    )
    .expect("rendu ColorMNet");
    let elapsed = started.elapsed().as_secs_f64();
    println!("ColorMNet : {outcome:?} en {elapsed:.1} s");
    if let Some(timings) = colorizer.timings() {
        println!("\n{}", timings.report());
        // Le temps de rendu inclut l'écriture des PNG, que le produit ne fait pas
        // (il alimente ffmpeg) : on sépare les deux pour ne pas optimiser à tort.
        let measured = timings.total_secs();
        println!(
            "  {:<34} {:8.1} ms/pas  {:5.1} %\n",
            "reste (E/S du harnais, ffmpeg, attente)",
            (elapsed - measured) * 1000.0 / timings.steps.max(1) as f64,
            (elapsed - measured) * 100.0 / elapsed.max(1e-9)
        );
    }
    extract_all(&output, &args.out.join("cm"));
}
