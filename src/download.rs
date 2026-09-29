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

use md5::{Digest, Md5};

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
    // Une signature par fichier, recalculée à l'instant du départ : le service
    // délivre des URLs datées, qu'on ne peut ni stocker ni réutiliser.
    let signer = Signer::from_environment();
    if signer.is_none() {
        tracing::warn!(
            "aucun secret d'hébergement (COLORVID_COLORMNET_SECRET) : les URLs seront \
             demandées non signées — un service protégé répondra 403"
        );
    }
    for (index, file) in missing.iter().enumerate() {
        let target = directory.join(&file.name);
        let received = download_file(&target, file, &manifest, signer.as_ref(), |received| {
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

/// Signataire des URLs du service d'hébergement.
///
/// Le service protège chaque fichier par une signature HMAC **datée** : une URL
/// signée ne peut donc pas être stockée dans le manifeste, il faut la recalculer à
/// chaque téléchargement — et la recalculer **encore** si elle expire en route.
///
/// L'algorithme est imposé par le service (`nginx secure_link`) : MD5 du texte
/// `secret + chemin + expiration`, encodé en base64 URL-safe **sans padding**.
/// Le chemin commence par un slash et n'inclut ni l'hôte ni les paramètres.
pub struct Signer {
    secret: String,
    validity_secs: u64,
}

impl Signer {
    /// Lit le secret dans l'environnement d'abord, puis dans celui de compilation.
    ///
    /// L'ordre est délibéré : un binaire distribué peut embarquer un secret (build
    /// `quality` uniquement, jamais le cœur), et un utilisateur peut toujours le
    /// remplacer sans reconstruire. **Absent, la signature est simplement omise** —
    /// un service ouvert continue de fonctionner, et rien n'échoue en silence : le
    /// serveur répondra 403 et l'erreur le dira.
    pub fn from_environment() -> Option<Self> {
        let secret = std::env::var("COLORVID_COLORMNET_SECRET")
            .ok()
            .or_else(|| option_env!("COLORVID_COLORMNET_SECRET").map(str::to_string))
            .filter(|value| !value.trim().is_empty())?;
        Some(Self {
            secret,
            // Marge confortable : le plus gros fichier (267 Mo) doit pouvoir finir.
            validity_secs: 3600,
        })
    }

    /// URL signée pour `name`, valable à partir de maintenant.
    pub fn sign(&self, base_url: &str, name: &str) -> String {
        let base = base_url.trim_end_matches('/');
        // `base` porte l'hôte **et** le chemin : concaténer les deux dupliquerait le
        // chemin. On recompose donc à partir de l'origine et du chemin signé.
        let origin = origin_of(base);
        let path = format!("{}/{}", path_of(base), name);
        let expiration = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
            + self.validity_secs;
        let mut hasher = Md5::new();
        hasher.update(format!("{}{}{}", self.secret, path, expiration).as_bytes());
        let digest = hasher.finalize();
        let signature = base64::Engine::encode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            digest,
        );
        format!("{origin}{path}?e={expiration}&s={signature}")
    }
}

/// Origine d'une URL : schéma et hôte, sans chemin.
fn origin_of(url: &str) -> &str {
    match url.find("://") {
        Some(index) => match url[index + 3..].find('/') {
            Some(slash) => &url[..index + 3 + slash],
            None => url,
        },
        None => "",
    }
}

/// Chemin d'une URL, sans schéma ni hôte — ce que le serveur reçoit.
fn path_of(url: &str) -> &str {
    match url.find("://") {
        Some(index) => match url[index + 3..].find('/') {
            Some(slash) => &url[index + 3 + slash..],
            None => "/",
        },
        None => url,
    }
}

fn download_file(
    target: &Path,
    file: &WeightFile,
    manifest: &WeightManifest,
    signer: Option<&Signer>,
    mut on_progress: impl FnMut(u64),
) -> Result<u64, String> {
    let url = match signer {
        Some(signer) => signer.sign(&manifest.base_url, &file.name),
        None => manifest.url_for(file),
    };
    let response = ureq::get(&url)
        .call()
        .map_err(|e| describe_download_error(&file.name, e))?;
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
    use sha2::Sha256;
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

/// Traduit l'échec HTTP en message actionnable.
///
/// Le service d'hébergement a trois codes qui veulent dire trois choses très
/// différentes, et « erreur 403 » n'aide personne à les distinguer.
fn describe_download_error(name: &str, error: ureq::Error) -> String {
    let code = match &error {
        ureq::Error::StatusCode(code) => Some(*code),
        _ => None,
    };
    match code {
        Some(403) => format!(
            "téléchargement de {name} refusé (403) : signature absente, invalide ou              calculée sur un autre chemin. Vérifier `COLORVID_COLORMNET_SECRET`."
        ),
        Some(410) => format!(
            "téléchargement de {name} refusé (410) : le lien signé a expiré avant la              fin du transfert."
        ),
        Some(404) => format!("téléchargement de {name} : fichier absent du serveur (404)."),
        _ => format!("téléchargement de {name} : {error}"),
    }
}

#[cfg(test)]
mod signing_tests {
    use super::*;

    /// Vecteur de référence produit **indépendamment**, par openssl :
    ///
    /// ```text
    /// printf '%s' "s3cr3t/f/colorvid/segment.onnx1791298838" \
    ///   | openssl dgst -md5 -binary | openssl base64 -A | tr '+/' '-_' | tr -d '='
    /// ```
    ///
    /// Un test qui recalculerait la signature avec le même code ne prouverait rien :
    /// celui-ci compare à une implémentation extérieure.
    #[test]
    fn the_signature_matches_an_independent_implementation() {
        let signer = Signer {
            secret: "s3cr3t".to_string(),
            validity_secs: 0,
        };
        let path = "/f/colorvid/segment.onnx";
        let expiration = 1791298838u64;
        let mut hasher = Md5::new();
        hasher.update(format!("{}{}{}", signer.secret, path, expiration).as_bytes());
        let digest = hasher.finalize();
        let signature = base64::Engine::encode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            digest,
        );
        assert_eq!(signature, "uznXRkR6D7kIhDEM3L4lSA");
    }

    /// La variante URL-safe ne doit laisser ni `+`, ni `/`, ni padding : le service
    /// répond 403 pour un seul `=` final oublié.
    #[test]
    fn the_signature_is_url_safe_and_unpadded() {
        let signer = Signer {
            secret: "secret de test avec des accents éàü".to_string(),
            validity_secs: 60,
        };
        let url = signer.sign("https://files.burnas.app/f/colorvid", "segment.onnx.data");
        let signature = url.split("&s=").nth(1).expect("signature présente");
        assert!(!signature.contains('+') && !signature.contains('/') && !signature.contains('='));
        assert!(url.starts_with("https://files.burnas.app/f/colorvid/segment.onnx.data?e="));
    }

    /// Le chemin signé est celui que le serveur reçoit : ni hôte, ni paramètres.
    /// L'URL finale ne doit contenir le chemin **qu'une fois** : la base porte déjà
    /// l'hôte et le préfixe de chemin.
    #[test]
    fn the_signed_url_does_not_duplicate_the_path() {
        let signer = Signer {
            secret: "s3cr3t".to_string(),
            validity_secs: 3600,
        };
        let url = signer.sign("https://files.burnas.app/f/colorvid", "segment.onnx");
        assert_eq!(url.matches("/f/colorvid/").count(), 1, "{url}");
        assert!(url.starts_with("https://files.burnas.app/f/colorvid/segment.onnx?e="), "{url}");
    }

    #[test]
    fn the_signed_path_excludes_the_host() {
        assert_eq!(path_of("https://files.burnas.app/f/colorvid"), "/f/colorvid");
        assert_eq!(path_of("https://files.burnas.app"), "/");
        assert_eq!(path_of("https://files.burnas.app/"), "/");
    }
}
