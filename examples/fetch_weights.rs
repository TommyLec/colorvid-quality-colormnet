//! Récupère les poids du moteur **sans lancer l'application**.
//!
//! Utile pour préparer une machine hors ligne, pour amorcer un cache, ou pour
//! diagnostiquer un téléchargement qui échoue côté application.
//!
//! ```text
//! COLORVID_COLORMNET_SECRET=… cargo run --release --example fetch_weights -- [dossier]
//! ```
//!
//! Sans argument, la destination est le dossier par défaut de l'application
//! (`$XDG_DATA_HOME/colorvid/colormnet` ou `~/.local/share/colorvid/colormnet`).

use colorvid_quality_colormnet::download::{self, WeightsOutcome};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = std::env::args()
        .nth(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(download::default_directory);
    println!("destination : {}", directory.display());

    // Le secret est lu par le module de téléchargement, dans l'environnement puis
    // dans celui de compilation. On le signale ici pour que l'absence de signature
    // ne soit jamais une surprise au moment d'un 403.
    match std::env::var("COLORVID_COLORMNET_SECRET") {
        Ok(value) if !value.trim().is_empty() => {
            println!("secret      : fourni par l'environnement ({} caractères)", value.len())
        }
        _ => println!(
            "secret      : ABSENT — les URLs seront demandées non signées ; un service \
             protégé répondra 403"
        ),
    }

    // Le dernier pourcentage affiché est suivi **par fichier** : sans cela, chaque
    // fichier se terminant à 100 %, les suivants n'afficheraient plus rien.
    let mut last: Option<(usize, u64)> = None;
    let outcome = download::ensure_weights(&directory, |progress| {
        let percent = (progress.received * 100)
            .checked_div(progress.expected)
            .unwrap_or(0);
        if last == Some((progress.index, percent)) {
            return;
        }
        last = Some((progress.index, percent));
        // L'unité s'adapte à la taille : les petits graphes font moins d'un mégaoctet,
        // et un arrondi en mégaoctets afficherait « 0 ».
        let received = progress.received as f64 / (1u64 << 20) as f64;
        let expected = progress.expected as f64 / (1u64 << 20) as f64;
        println!(
            "  [{}/{}] {} — {} % ({received:.1}/{expected:.1} Mo)",
            progress.index + 1,
            progress.total,
            progress.file,
            percent
        );
    })?;

    match outcome {
        WeightsOutcome::AlreadyPresent => {
            println!("déjà présents et conformes : aucun téléchargement");
        }
        WeightsOutcome::Downloaded { files, bytes } => {
            println!(
                "téléchargés et vérifiés : {files} fichiers, {} Mo",
                bytes / (1 << 20)
            );
        }
    }
    println!(
        "les poids sont exploitables : {}",
        download::weights_are_present(&directory)
    );
    Ok(())
}
