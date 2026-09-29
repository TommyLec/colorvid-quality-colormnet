//! # Moteur « quality » ColorMNet — crate isolée (dérogation CC BY-NC-SA 4.0)
//!
//! **Toute** la dérogation vit dans cette crate : aucun type, aucune constante et
//! aucune fonction n'est partagée avec le moteur par défaut au-delà de
//! l'abstraction `ImageColorizer` d'`ai-core`. Le retrait consiste à supprimer ce
//! dossier, la dépendance optionnelle de `app/` et la feature Cargo — cf.
//! `docs/plan-colormnet-quality.md` §9.
//!
//! ## Contenu
//! - [`state`] — l'**état récurrent** externalisé (calendrier de pas, mémoire
//!   vive, clé/valeur courts termes) et sa politique de rétention ;
//! - [`store`] — la **mémoire clé/valeur** et le suivi par objet ;
//! - [`memory`] — la lecture mémoire (similarité, top-k, agrégation) ;
//! - [`input`] — la préparation des entrées (Lab, exemplaire, complétion 112) ;
//! - [`geometry`] — l'alignement sur 112 ;
//! - [`engine`] — l'**orchestration des quatre graphes** ONNX.
//!
//! ## Ce qui n'est pas fait
//! Le branchement dans le rendu batch (exemplaire par plan, `reset_sequence` aux
//! coupes, désactivation des étages DDColor) et la sélection dans l'UI relèvent
//! des jalons suivants (`docs/plan-colormnet-quality.md` §13.5).
//!
//! ⚠️ Licence : portions adaptées de ColorMNet (CC BY-NC-SA 4.0). Voir `NOTICE.md`.

pub mod engine;
pub mod geometry;
pub mod input;
pub mod memory;
pub mod state;
pub mod catalog;
pub mod download;
pub mod store;

use colorvid_ai_core::error::AiCoreError;
use colorvid_ai_core::model::verify_checksum;

pub use catalog::{ATTRIBUTION, DISPLAY_NAME, LICENSE};
pub use engine::{ColorMNetEngine, Shapes, CHROMA_OBJECTS};

/// Résolution spatiale des graphes exportés (`§11` du plan). Les quatre graphes
/// sont figés à cette taille : le moteur dérive sa résolution de travail pour que
/// la complétion y retombe exactement.
pub const SUPPORTED_WIDTH: u32 = 640;
pub const SUPPORTED_HEIGHT: u32 = 360;

/// Noms des graphes exportés (jalon C2). Les `.onnx.data` éventuels doivent
/// rester à côté du `.onnx` correspondant.
pub const GRAPH_FILES: [&str; 4] = [
    "encode_key.onnx",
    "short_term_attn.onnx",
    "segment.onnx",
    "encode_value.onnx",
];

/// Erreurs propres au moteur « quality ».
#[derive(Debug, thiserror::Error)]
pub enum QualityError {
    #[error("graphe ONNX manquant : {0}")]
    MissingGraph(String),
    #[error("état récurrent : {0}")]
    State(#[from] crate::store::StoreError),
    #[error(
        "exemplaire manquant : ColorMNet est un moteur à référence, la première frame du plan \
         doit recevoir une image-exemplaire en couleur"
    )]
    MissingExemplar,
}

/// Chemins des quatre graphes et de leurs checksums.
#[derive(Debug, Clone)]
pub struct GraphSet {
    pub directory: std::path::PathBuf,
}

impl GraphSet {
    pub fn new(directory: impl Into<std::path::PathBuf>) -> Self {
        Self {
            directory: directory.into(),
        }
    }

    pub fn graph_path(&self, file: &str) -> String {
        self.directory.join(file).display().to_string()
    }

    pub fn checksum_path(&self, file: &str) -> String {
        self.directory.join(format!("{file}.sha256")).display().to_string()
    }

    /// Vérifie l'intégrité des quatre graphes **avant** tout chargement.
    /// Les empreintes attendues sont ancrées dans le catalogue de modèles
    /// (jalon C6) ; ici on vérifie au minimum la cohérence fichier/checksum.
    pub fn verify(&self, anchored: &[(&str, &str)]) -> Result<(), AiCoreError> {
        for file in GRAPH_FILES {
            let path = self.graph_path(file);
            if !std::path::Path::new(&path).exists() {
                return Err(AiCoreError::ModelNotFound(format!(
                    "{file} absent de {}",
                    self.directory.display()
                )));
            }
            let expected = anchored
                .iter()
                .find(|(name, _)| *name == file)
                .map(|(_, sha)| *sha);
            verify_checksum(&path, &self.checksum_path(file), expected)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_set_paths_are_derived_from_the_directory() {
        let set = GraphSet::new("/tmp/quality");
        assert_eq!(set.graph_path("segment.onnx"), "/tmp/quality/segment.onnx");
        assert_eq!(
            set.checksum_path("segment.onnx"),
            "/tmp/quality/segment.onnx.sha256"
        );
        assert_eq!(GRAPH_FILES.len(), 4);
    }

    #[test]
    fn verify_reports_a_missing_graph_instead_of_panicking() {
        let set = GraphSet::new("/tmp/colorvid-quality-inexistant");
        let err = set.verify(&[]).unwrap_err();
        assert!(matches!(err, AiCoreError::ModelNotFound(_)), "{err}");
    }

    #[test]
    fn engine_config_defaults_match_the_python_configuration() {
        let cfg = state::EngineConfig::default();
        assert_eq!(cfg.mem_every, 5);
        assert_eq!(cfg.deep_update_every, -1);
        assert!(cfg.deep_update_sync());
        assert_eq!(cfg.top_k, 30);
        assert_eq!(cfg.window_frames(), Some(10), "T_max de la configuration amont");
    }
}
