//! Téléchargement des poids à la demande — **jamais dans l'installeur**.
//!
//! `AGENTS.md` l'impose pour la dérogation ColorMNet (CC BY-NC-SA 4.0) : les poids
//! sont récupérés au premier usage, **ancrés par empreinte**, accompagnés du texte
//! de licence et de la provenance. Hors ligne, l'erreur doit être claire — v1 ne
//! prévoit pas de reprise.
//!
//! Ce module vit dans la crate dédiée : le client HTTP n'existe pas dans le binaire
//! par défaut, et retirer la dérogation l'emporte avec le reste (§9 du plan).

use std::io::Read;
use std::path::{Path, PathBuf};

use crate::catalog;

/// Une entrée du manifeste : un fichier de poids, son empreinte et sa taille.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct WeightFile {
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
}

/// Manifeste des poids : ce qui doit être présent pour que le moteur tourne.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct WeightManifest {
    /// Base des téléchargements ; `{name}` est remplacé par le nom du fichier.
    pub base_url: String,
    pub source: String,
    pub license: String,
    pub files: Vec<WeightFile>,
}

impl WeightManifest {
    /// Manifeste livré avec le binaire, empreintes incluses.
    pub fn bundled() -> Result<Self, String> {
        serde_json::from_str(include_str!("../assets/weights.json"))
            .map_err(|e| format!("manifeste des poids illisible : {e}"))
    }

    /// URL effective d'un fichier. `COLORVID_COLORMNET_BASE_URL` remplace la base
    /// livrée — utile pour un miroir d'entreprise ou un hébergement local.
    pub fn url_for(&self, file: &WeightFile) -> String {
        let base = std::env::var(catalog::BASE_URL_ENV)
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| self.base_url.clone());
        format!("{}/{}", base.trim_end_matches('/'), file.name)
    }
}

/// Avancement du téléchargement, en octets.
pub struct DownloadProgress<'a> {
    pub file: &'a str,
    pub index: usize,
    pub total: usize,
    pub received: u64,
    pub expected: u64,
}

/// Résultat d'une mise en place des poids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightsOutcome {
    /// Tous les fichiers étaient déjà présents et vérifiés.
    AlreadyPresent,
    /// Certains fichiers ont été téléchargés.
    Downloaded { files: usize, bytes: u64 },
}

/// S'assure que les poids sont présents et conformes dans `directory`.
///
/// Un fichier présent dont l'empreinte ne correspond pas est **retéléchargé** : une
/// interruption laisse un `.part` qui n'est jamais pris pour un fichier valide.
pub fn ensure_weights(
    directory: &Path,
    mut progress: impl FnMut(DownloadProgress<'_>),
) -> Result<WeightsOutcome, String> {
    let manifest = WeightManifest::bundled()?;
    std::fs::create_dir_all(directory)
        .map_err(|e| format!("dossier des poids {} : {e}", directory.display()))?;

    let mut missing = Vec::new();
    for file in &manifest.files {
        if verify_file(&directory.join(&file.name), file).is_ok() {
            continue;
        }
        missing.push(file);
    }
    if missing.is_empty() {
        write_notices(directory, &manifest)?;
        return Ok(WeightsOutcome::AlreadyPresent);
    }

    let total = missing.len();
    let mut bytes = 0u64;
    for (index, file) in missing.iter().enumerate() {
        let target = directory.join(&file.name);
        let url = manifest.url_for(file);
        let received = download_file(&url, &target, file, |received| {
            progress(DownloadProgress {
                file: &file.name,
                index,
                total,
                received,
                expected: file.bytes,
            });
        })?;
        bytes += received;
    }
    write_notices(directory, &manifest)?;
    Ok(WeightsOutcome::Downloaded {
        files: total,
        bytes,
    })
}

/// Vérifie un fichier présent : taille **et** empreinte SHA-256.
///
/// Exporté pour les tests et pour l'app, qui peut ainsi expliquer une installation
/// incomplète sans rien télécharger.
pub fn verify_file(path: &Path, file: &WeightFile) -> Result<(), String> {
    let metadata = std::fs::metadata(path)
        .map_err(|e| format!("{} introuvable : {e}", path.display()))?;
    if metadata.len() != file.bytes {
        return Err(format!(
            "{} : {} octets au lieu de {}",
            path.display(),
            metadata.len(),
            file.bytes
        ));
    }
    let digest = sha256_file(path)?;
    if digest != file.sha256 {
        return Err(format!("{} : empreinte {digest} ≠ {}", path.display(), file.sha256));
    }
    Ok(())
}

/// Vérifie que **tous** les poids sont en place, sans réseau.
pub fn weights_are_present(directory: &Path) -> bool {
    let Ok(manifest) = WeightManifest::bundled() else {
        return false;
    };
    manifest
        .files
        .iter()
        .all(|file| verify_file(&directory.join(&file.name), file).is_ok())
}

fn download_file(
    url: &str,
    target: &Path,
    file: &WeightFile,
    mut on_progress: impl FnMut(u64),
) -> Result<u64, String> {
    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("téléchargement de {} : {e}", file.name))?;
    let mut reader = response.into_body().into_reader();

    // Écriture dans un `.part` : un fichier tronqué ne peut pas être pris pour un
    // poids valide si l'utilisateur interrompt le téléchargement.
    let part = target.with_extension(format!(
        "{}part",
        target.extension().and_then(|e| e.to_str()).map(|e| format!("{e}.")).unwrap_or_default()
    ));
    let mut sink = std::fs::File::create(&part)
        .map_err(|e| format!("création de {} : {e}", part.display()))?;
    let mut buffer = vec![0u8; 1 << 20];
    let mut received = 0u64;
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|e| format!("lecture du flux {} : {e}", file.name))?;
        if read == 0 {
            break;
        }
        std::io::Write::write_all(&mut sink, &buffer[..read])
            .map_err(|e| format!("écriture de {} : {e}", part.display()))?;
        received += read as u64;
    }
    drop(sink);
    on_progress(received);

    verify_file(&part, file).map_err(|e| {
        let _ = std::fs::remove_file(&part);
        format!("{e} — téléchargement corrompu, réessayez")
    })?;
    std::fs::rename(&part, target)
        .map_err(|e| format!("installation de {} : {e}", target.display()))?;
    Ok(received)
}

fn write_notices(directory: &Path, manifest: &WeightManifest) -> Result<(), String> {
    let notice = format!(
        "Poids du moteur « quality » ColorMNet\n\
         ====================================\n\n\
         Source : {}\n\
         Licence : {}\n\n\
         Ces poids ne sont pas distribués avec l'application. Ils sont téléchargés à\n\
         la demande et vérifiés par empreinte SHA-256 avant usage.\n\n\
         Attribution : ColorMNet (ECCV 2024) — Yang, Dong, Tang, Pan.\n\
         Modèle adapté et exporté en ONNX pour ColorVid.\n\n\
         Œuvres dont ColorMNet dérive, créditées comme l'exige la licence :\n\
           - XMem — https://github.com/hkchengrex/XMem (MIT)\n\
           - DINOv2 — https://github.com/facebookresearch/dinov2 (Apache-2.0)\n\
           - projet amont — https://github.com/yyang181/colormnet\n\n\
         Empreintes :\n{}",
        manifest.source,
        manifest.license,
        manifest
            .files
            .iter()
            .map(|f| format!("  {}  {}  {}", f.sha256, f.bytes, f.name))
            .collect::<Vec<_>>()
            .join("\n")
    );
    std::fs::write(directory.join("NOTICE.txt"), notice)
        .map_err(|e| format!("écriture du NOTICE : {e}"))?;
    std::fs::write(
        directory.join("LICENSE-CC-BY-NC-SA-4.0.txt"),
        include_str!("../assets/LICENSE-CC-BY-NC-SA-4.0.txt"),
    )
    .map_err(|e| format!("écriture de la licence : {e}"))?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("ouverture de {} : {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|e| format!("lecture de {} : {e}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Répertoire par défaut des poids, créé au besoin.
pub fn default_directory() -> PathBuf {
    if let Some(directory) = std::env::var_os(catalog::GRAPHS_ENV) {
        return PathBuf::from(directory);
    }
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local").join("share"))
        })
        .unwrap_or_else(std::env::temp_dir);
    base.join("colorvid").join("colormnet")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> WeightManifest {
        WeightManifest::bundled().expect("manifeste livré")
    }

    /// Le manifeste doit couvrir les quatre graphes **et** leurs données externes.
    ///
    /// Garde explicite, et non déduite : la première version listait sept fichiers
    /// en omettant `short_term_attn.onnx.data`, apparu quand ce graphe a été
    /// ré-exporté avec des tranches statiques (C2quinquies). Un testeur aurait
    /// téléchargé un graphe amputé. Le manifeste est désormais **engendré** depuis
    /// le dossier d'export (`.j0/colormnet/c2sexies_manifest.py`) ; ce test épingle
    /// le résultat attendu pour qu'un export modifié ne passe pas inaperçu.
    #[test]
    fn the_manifest_covers_every_graph_and_its_external_data() {
        const EXPECTED: [&str; 8] = [
            "encode_key.onnx",
            "encode_key.onnx.data",
            "short_term_attn.onnx",
            "short_term_attn.onnx.data",
            "segment.onnx",
            "segment.onnx.data",
            "encode_value.onnx",
            "encode_value.onnx.data",
        ];
        let manifest = manifest();
        let mut names: Vec<&str> = manifest.files.iter().map(|f| f.name.as_str()).collect();
        names.sort_unstable();
        let mut expected = EXPECTED.to_vec();
        expected.sort_unstable();
        assert_eq!(
            names, expected,
            "le manifeste ne correspond plus à l'export — le régénérer avec \
             .j0/colormnet/c2sexies_manifest.py, puis reporter la liste attendue"
        );
        assert!(manifest.license.contains("NC"), "licence non commerciale");
        assert!(!manifest.source.is_empty(), "provenance exigée");
        let total: u64 = manifest.files.iter().map(|f| f.bytes).sum();
        assert!(
            (400..800).contains(&(total / 1024 / 1024)),
            "poids suspects : {} Mo au total",
            total / 1024 / 1024
        );
    }

    #[test]
    fn a_wrong_size_or_digest_is_rejected() {
        let directory = std::env::temp_dir().join("colorvid-poids-test");
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let file = WeightFile {
            name: "petit.bin".into(),
            bytes: 4,
            sha256: "00".repeat(32),
        };
        let path = directory.join("petit.bin");
        std::fs::write(&path, b"abcd").unwrap();
        // taille juste, empreinte fausse
        assert!(verify_file(&path, &file).is_err());
        // taille fausse
        std::fs::write(&path, b"abc").unwrap();
        assert!(verify_file(&path, &file).unwrap_err().contains("3 octets"));
        std::fs::remove_dir_all(&directory).ok();
    }

    /// La base peut être redirigée (miroir local) : c'est ce qui rend le
    /// téléchargement testable et déployable hors d'un hébergement public.
    #[test]
    fn the_base_url_can_be_redirected() {
        let manifest = manifest();
        let file = &manifest.files[0];
        std::env::set_var(catalog::BASE_URL_ENV, "https://miroir.exemple/poids/");
        assert_eq!(
            manifest.url_for(file),
            format!("https://miroir.exemple/poids/{}", file.name)
        );
        std::env::remove_var(catalog::BASE_URL_ENV);
        assert!(manifest.url_for(file).starts_with(&manifest.base_url));
    }
}
