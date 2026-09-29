//! Catalogue et résolution des poids — **entièrement dans le module dédié**.
//!
//! Le registre partagé (`ai-core`) ignore tout de ColorMNet : le nom d'affichage,
//! la licence et la résolution des graphes vivent ici. C'est la règle d'isolation
//! n° 1 d'`AGENTS.md` — aucun type, aucune constante partagés avec le moteur
//! DDColor au-delà de l'abstraction `ImageColorizer`.
//!
//! Conséquence voulue : **retirer la dérogation** se limite à supprimer ce module,
//! la feature Cargo et le variant de sélection, sans toucher au registre commun.

use std::path::{Path, PathBuf};

use colorvid_ai_core::error::AiCoreError;
use colorvid_types::config::ModelCapabilities;

use crate::GraphSet;

/// Identifiant du moteur, choisi par le greffon.
pub const MODEL_ID: &str = "colorMNetQuality";

/// Nom affiché dans le sélecteur de moteur.
pub const DISPLAY_NAME: &str = "ColorMNet (qualité)";

/// Licence du modèle amont (ECCV 2024, Yang, Dong, Tang, Pan) : **non commerciale
/// et partage à l'identique**. Un binaire qui embarque ce moteur est un binaire non
/// commercial ; toute partie adaptée (notre export ONNX) reste sous cette licence.
pub const LICENSE: &str = "CC BY-NC-SA 4.0";

/// Attribution à afficher (écran « À propos » et sélecteur de moteur).
///
/// La CC BY-NC-SA impose de créditer l'œuvre **et** celles dont elle dérive : le
/// `LICENSES` amont déclare XMem (MIT) et DINOv2 (Apache-2.0). Les omettre serait
/// une attribution incomplète.
pub const ATTRIBUTION: &str = "ColorMNet (ECCV 2024) — Yang, Dong, Tang, Pan. \
    Modèle adapté et exporté en ONNX pour ColorVid. Licence CC BY-NC-SA 4.0, \
    usage non commercial. Dérive de XMem (MIT) et emprunte à DINOv2 (Apache-2.0).";

/// Taille des quatre graphes et de leurs données externes, mesurée sur l'export C2.
pub const ESTIMATED_SIZE_MB: u64 = 475;

/// Variable d'environnement qui pointe le dossier des graphes (packaging C6 :
/// poids téléchargés à la demande, jamais dans l'installeur).
pub const GRAPHS_ENV: &str = "COLORVID_COLORMNET_GRAPHS";

/// Redirige la base de téléchargement (miroir d'entreprise, hébergement local).
pub const BASE_URL_ENV: &str = "COLORVID_COLORMNET_BASE_URL";

/// Dossier de développement, à la racine du dépôt.
const DEV_DIRECTORY: &str = ".j0/colormnet/onnx";

/// Résout le dossier des graphes, ou explique précisément ce qui manque.
///
/// Ordre : `COLORVID_COLORMNET_GRAPHS`, puis le dossier de développement du dépôt.
/// En distribution, c'est le premier qui compte — et un dossier absent **ne
/// retombe pas** sur un autre moteur : il produit une erreur lisible.
pub fn resolve_graph_set() -> Result<GraphSet, AiCoreError> {
    if let Some(directory) = std::env::var_os(GRAPHS_ENV) {
        let set = GraphSet::new(PathBuf::from(directory));
        return set
            .verify(&[])
            .map(|()| set)
            .map_err(|e| AiCoreError::ModelNotFound(format!("{GRAPHS_ENV} : {e}")));
    }
    // Dossier des poids téléchargés à la demande (le cas normal en distribution).
    let downloaded = crate::download::default_directory();
    if crate::download::weights_are_present(&downloaded) {
        return Ok(GraphSet::new(downloaded));
    }
    let root = workspace_root();
    let set = GraphSet::new(root.join(DEV_DIRECTORY));
    set.verify(&[]).map(|()| set).map_err(|_| {
        AiCoreError::ModelNotFound(format!(
            "graphes ColorMNet absents. Téléchargement à la demande (jalon C6) ou, en \
             développement, les placer dans {} / définir {GRAPHS_ENV}",
            root.join(DEV_DIRECTORY).display()
        ))
    })
}

fn workspace_root() -> PathBuf {
    // `CARGO_MANIFEST_DIR` = crates/quality-colormnet → racine du dépôt.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .unwrap_or(Path::new("."))
        .to_path_buf()
}

/// Les poids sont-ils **déjà là et vérifiés**, sans réseau ?
///
/// Distinct de la disponibilité du moteur : le moteur est proposable dès que le
/// binaire l'embarque, et les poids se téléchargent au premier rendu.
pub fn weights_present() -> bool {
    crate::download::weights_are_present(&crate::download::default_directory())
        || resolve_graph_set().is_ok()
}

/// Entrée de catalogue, au format attendu par l'IPC.
pub fn capabilities() -> ModelCapabilities {
    ModelCapabilities {
        // L'identifiant appartient au greffon : le cœur ne le connaît pas.
        id: MODEL_ID.to_string(),
        display_name: DISPLAY_NAME.to_string(),
        fixed_width: crate::SUPPORTED_WIDTH,
        fixed_height: crate::SUPPORTED_HEIGHT,
        input_format: "grayscale RGB".to_string(),
        output_format: "CIELAB ab".to_string(),
        gpu_required: true,
        estimated_size_mb: ESTIMATED_SIZE_MB,
        // Le moteur **exige** une image-exemplaire par plan : sans elle, il refuse
        // de rendre (aucune couleur inventée).
        supports_reference: true,
        // Jamais dans l'installeur : licence NC-SA, téléchargement à la demande.
        bundled_and_verified: false,
        // Le moteur est proposable dès que ce binaire l'embarque : les poids se
        // téléchargent à la demande, au premier rendu, avec l'avancement affiché.
        // Masquer le moteur tant que les poids manquent rendrait le téléchargement
        // inatteignable — l'utilisateur ne pourrait jamais le déclencher.
        available: true,
        attribution: Some(ATTRIBUTION.to_string()),
        license: LICENSE.to_string(),
        // ColorMNet est un moteur **à exemplaire** : sans référence par plan, il
        // refuse de rendre (voir `build_anchors`).
        requires_reference: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_catalog_entry_declares_its_licence_constraints() {
        let entry = capabilities();
        assert_eq!(entry.id, MODEL_ID);
        // Les deux garanties de la dérogation, inscrites dans le catalogue.
        assert!(!entry.bundled_and_verified, "jamais dans l'installeur");
        assert!(entry.supports_reference, "moteur à exemplaire");
        assert!(LICENSE.contains("NC"), "licence non commerciale");
        assert!(ATTRIBUTION.contains("ECCV 2024"), "attribution présente");
    }

    #[test]
    fn an_absent_directory_is_an_explicit_error_not_a_fallback() {
        // Un dossier vide ne doit pas « réussir » : l'appelant doit pouvoir
        // refuser le rendu avec un message clair (règle v1 : pas de repli muet).
        let set = GraphSet::new(std::env::temp_dir().join("colorvid-absent-graphes"));
        assert!(set.verify(&[]).is_err());
    }
}
