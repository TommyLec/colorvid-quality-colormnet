//! Sonde **graphe par graphe** sur données **réelles**.
//!
//! Le smoke test MIGraphX à entrées factices passe sur les quatre graphes, mais le
//! rendu réel plante (faut mémoire GPU allant jusqu'au reset de la carte). Cette
//! sonde isole le coupable : **un seul graphe, un seul processus, de vraies
//! entrées** capturées par `.j0/colormnet/c2bis_real_inputs.py`.
//!
//! ⚠️ À exécuter **hors session graphique** (console virtuelle) : un faut GPU peut
//! réinitialiser la carte et emporter le compositeur.
//!
//! ```bash
//! # une commande par graphe, en commençant par le plus suspect
//! cargo run --release -p colorvid-quality-colormnet --example graph_probe \
//!   --features colorvid-ai-core/migraphx,colorvid-ai-core/load-dynamic -- \
//!   short_term_attn .j0/colormnet/realdata/short_term_attn --provider migraphx
//! ```

use std::path::{Path, PathBuf};

use colorvid_ai_core::provider::{build_session, ProviderChoice};
use colorvid_types::config::GpuProvider;
use ort::value::Tensor;

fn provider_label(provider: &ProviderChoice) -> &'static str {
    match provider {
        ProviderChoice::Cpu => "cpu",
        ProviderChoice::Gpu(_) => "gpu",
    }
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

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();
    colorvid_ai_core::init_ort_environment();

    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.len() < 2 {
        eprintln!("usage: graph_probe <graphe> <dossier_entrées> [--provider cpu|migraphx]");
        std::process::exit(2);
    }
    let graph = raw[0].clone();
    let inputs = PathBuf::from(&raw[1]);
    let provider = match raw
        .iter()
        .position(|a| a == "--provider")
        .and_then(|p| raw.get(p + 1))
        .map(String::as_str)
    {
        Some("migraphx") => ProviderChoice::Gpu(GpuProvider::Migraphx),
        _ => ProviderChoice::Cpu,
    };

    let model = inputs.join(format!("{graph}.onnx"));
    // Sortie brute : permet de comparer les tenseurs exacts entre providers (`cmp`).
    let dump = raw
        .iter()
        .position(|a| a == "--dump")
        .and_then(|p| raw.get(p + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| inputs.join(format!("out_{}.bin", provider_label(&provider))));
    println!("graphe {graph} | modèle {} | provider {provider:?}", model.display());

    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(inputs.join("manifest.json")).unwrap())
            .expect("manifest.json");
    let mut tensors = Vec::new();
    let mut names = Vec::new();
    for entry in manifest.as_array().expect("manifeste") {
        let name = entry["name"].as_str().unwrap().to_string();
        let shape: Vec<usize> = entry["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_u64().unwrap() as usize)
            .collect();
        let data = read_f32(&inputs.join(entry["file"].as_str().unwrap()));
        println!("  entrée {name:16} {shape:?} ({} valeurs)", data.len());
        tensors.push((shape, data));
        names.push(name);
    }

    let started = std::time::Instant::now();
    let mut session = build_session(&model.display().to_string(), provider, Some(8))
        .unwrap_or_else(|e| panic!("création de session : {e}"));
    println!("session créée en {:.1} s", started.elapsed().as_secs_f64());

    let inputs: Vec<(&str, Tensor<f32>)> = names
        .iter()
        .zip(tensors.iter())
        .map(|(name, (shape, data))| {
            let tensor = Tensor::from_array((shape.clone(), data.clone()))
                .unwrap_or_else(|e| panic!("tenseur {name} : {e}"));
            (name.as_str(), tensor)
        })
        .collect();

    let started = std::time::Instant::now();
    let outputs = session
        .run(inputs)
        .unwrap_or_else(|e| panic!("exécution : {e}"));
    println!("inférence en {:.3} s", started.elapsed().as_secs_f64());
    // Signature volontairement sensible : deux implémentations qui donnent la même
    // moyenne peuvent différer ailleurs. Le dump brut permet de comparer les
    // tenseurs exacts entre deux providers (`cmp` des .bin).
    let mut all = Vec::new();
    // Index des sorties : permet de découper le `.bin` concaténé pour comparer
    // **tenseur par tenseur** entre deux providers (bisection d'un graphe).
    let mut index = Vec::new();
    for (name, value) in outputs.iter() {
        let (shape, data) = value
            .try_extract_tensor::<f32>()
            .unwrap_or_else(|e| panic!("extraction de {name} : {e}"));
        let abs_mean = data.iter().map(|v| v.abs()).sum::<f32>() / data.len().max(1) as f32;
        let max = data.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        // somme en f64 : un écart d'arrondi la déplace, une moyenne non.
        let sum64: f64 = data.iter().map(|v| *v as f64).sum();
        let head: Vec<String> = data.iter().take(4).map(|v| format!("{v:.6}")).collect();
        println!(
            "  sortie {name:8} {shape:?} | |x| moyen {abs_mean:.6} max {max:.6} | \
             somme {sum64:.6} | début [{}]",
            head.join(", ")
        );
        index.push(serde_json::json!({
            "name": name,
            "shape": shape.to_vec(),
            "len": data.len(),
        }));
        all.extend_from_slice(data);
    }
    std::fs::write(&dump, all.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())
        .unwrap_or_else(|e| panic!("écriture de {} : {e}", dump.display()));
    let index_path = dump.with_extension("json");
    std::fs::write(
        &index_path,
        serde_json::to_string_pretty(&serde_json::Value::Array(index)).unwrap(),
    )
    .unwrap_or_else(|e| panic!("écriture de {} : {e}", index_path.display()));
    println!("  sorties brutes → {} (+ index .json)", dump.display());
    println!("GRAPHE {graph} : OK ✓ (provider {provider:?})");
}
