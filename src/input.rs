//! Préparation des entrées du réseau — ce que la chaîne amont fait avant
//! `InferenceCore.step`.
//!
//! Trois étapes, reproduites telles quelles :
//!
//! 1. **résolution de travail** — les graphes exportés (C2) ont des formes
//!    **statiques** `[1, 3, 448, 672]` ; la frame est donc ramenée à la
//!    résolution de travail [`WORK_WIDTH`]×[`WORK_HEIGHT`], comme le moteur par
//!    défaut ramène à 512×512 ;
//! 2. **canal L** — `skimage.color.rgb2lab` puis `(L - 50) / 50`, répliqué sur
//!    3 canaux (`im_rgb2lab_normalization`, `img_lll`) ;
//! 3. **exemplaire** — les canaux `ab` de l'image de référence, normalisés
//!    `a / 110`, `b / 110` (`msk[:, 1:3]`).
//!
//! Puis complétion par zéros au multiple de 112 (`pad_divide_by`) : c'est la
//! seule partie déjà présente dans [`crate::geometry`].
//!
//! ## Pourquoi une conversion Lab locale
//!
//! `AGENTS.md` interdit de partager du code avec le moteur par défaut au-delà de
//! l'abstraction `ImageColorizer` : cette crate ne dépend donc **pas** de
//! `colorvid-media`, et les quelques lignes de conversion sont réécrites ici,
//! avec les constantes **exactes de `skimage`** (matrice sRGB et point blanc
//! D65/2°) pour coller à la référence Python plutôt qu'à OpenCV.

use image::{imageops, RgbImage};

use crate::geometry::{pad_amounts, PadAmounts, PAD_MULTIPLE};

/// Résolution de travail : la plus grande qui **remplit la forme complétée des
/// graphes** (672×448 pour l'export C2) en **préservant le rapport d'aspect de
/// la source**.
///
/// Les graphes ont une forme statique de 3:2. Travailler à une résolution fixe
/// (16:9, celle du premier extrait de référence) **étirerait** toute source d'un
/// autre rapport — une source 4:3 de 33 % — et le modèle verrait une image
/// déformée. On dérive donc la taille de la source, avec la seule contrainte que
/// [`pad_amounts`] retombe exactement sur la forme des graphes : la dimension
/// doit tomber dans `(forme - 112, forme]`. En dehors de cette fenêtre (sources
/// très petites), on borne — la déformation résiduelle est alors le prix des
/// formes statiques, et elle est documentée.
pub fn work_size(
    source_width: u32,
    source_height: u32,
    padded_width: usize,
    padded_height: usize,
) -> (usize, usize) {
    let (source_width, source_height) = (source_width.max(1), source_height.max(1));
    // Cas nominal : la source est déjà dans la fenêtre qui complète exactement
    // sur la forme des graphes — on la garde **telle quelle**, sans
    // rééchantillonnage. C'est le comportement de l'amont (`--size -1`), et c'est
    // ce qui rend le portage comparable à la référence Python.
    let inside = |value: u32, padded: usize| {
        value as usize > padded.saturating_sub(PAD_MULTIPLE) && value as usize <= padded
    };
    if inside(source_width, padded_width) && inside(source_height, padded_height) {
        return (source_width as usize, source_height as usize);
    }
    let (sw, sh) = (source_width as f64, source_height as f64);
    let scale = (padded_width as f64 / sw).min(padded_height as f64 / sh);
    let fit = |value: f64, padded: usize| -> usize {
        let target = (value * scale).round().max(1.0) as usize;
        // La complétion doit retomber exactement sur la forme des graphes.
        target.clamp(padded.saturating_sub(PAD_MULTIPLE - 1), padded)
    };
    (fit(sw, padded_width), fit(sh, padded_height))
}

/// Matrice sRGB → XYZ (`skimage.color.colorconv.xyz_from_rgb`).
const XYZ_FROM_RGB: [[f32; 3]; 3] = [
    [0.412_453, 0.357_580, 0.180_423],
    [0.212_671, 0.715_160, 0.072_169],
    [0.019_334, 0.119_193, 0.950_227],
];

/// Point blanc D65 / observateur 2° (`skimage`, `_illuminants["D65"]["2"]`).
const WHITE_D65: [f32; 3] = [0.950_47, 1.0, 1.088_83];

/// Conversion sRGB (0-255) → CIELAB, identique à `skimage.color.rgb2lab`.
pub fn rgb_to_lab(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let lin = |v: u8| {
        let x = v as f32 / 255.0;
        if x > 0.040_45 {
            ((x + 0.055) / 1.055).powf(2.4)
        } else {
            x / 12.92
        }
    };
    let (r, g, b) = (lin(r), lin(g), lin(b));
    let mut xyz = [0.0f32; 3];
    for (row, out) in XYZ_FROM_RGB.iter().zip(xyz.iter_mut()) {
        *out = row[0] * r + row[1] * g + row[2] * b;
    }
    let f = |i: usize| {
        let t = xyz[i] / WHITE_D65[i];
        if t > 0.008_856 {
            t.cbrt()
        } else {
            7.787 * t + 16.0 / 116.0
        }
    };
    let (fx, fy, fz) = (f(0), f(1), f(2));
    (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
}

/// Complète un tenseur `[c, h, w]` par des zéros au multiple de 112.
pub fn pad_chw(src: &[f32], channels: usize, width: usize, height: usize, pad: PadAmounts) -> Vec<f32> {
    assert_eq!(src.len(), channels * width * height);
    if pad.is_identity() {
        return src.to_vec();
    }
    let (pw, ph) = pad.padded_size(width, height);
    let mut out = vec![0.0f32; channels * pw * ph];
    for c in 0..channels {
        for y in 0..height {
            let dst = c * pw * ph + (y + pad.top) * pw + pad.left;
            let src_row = c * width * height + y * width;
            out[dst..dst + width].copy_from_slice(&src[src_row..src_row + width]);
        }
    }
    out
}

/// Recadre un tenseur `[c, ph, pw]` complété en `[c, h, w]`.
pub fn unpad_chw(
    src: &[f32],
    channels: usize,
    width: usize,
    height: usize,
    pad: PadAmounts,
) -> Vec<f32> {
    let (pw, ph) = pad.padded_size(width, height);
    assert_eq!(src.len(), channels * pw * ph);
    if pad.is_identity() {
        return src.to_vec();
    }
    let mut out = vec![0.0f32; channels * width * height];
    for c in 0..channels {
        for y in 0..height {
            let src_row = c * pw * ph + (y + pad.top) * pw + pad.left;
            let dst = c * width * height + y * width;
            out[dst..dst + width].copy_from_slice(&src[src_row..src_row + width]);
        }
    }
    out
}

/// Entrées prêtes pour un pas d'inférence.
#[derive(Debug, Clone)]
pub struct PreparedStep {
    /// `[1, 3, ph, pw]` — canal L normalisé, répliqué.
    pub image: Vec<f32>,
    /// `[1, 2, ph, pw]` — `ab` de l'exemplaire normalisés, si fourni.
    pub exemplar: Option<Vec<f32>>,
    /// Résolution de travail (non complétée).
    pub width: usize,
    pub height: usize,
    pub pad: PadAmounts,
}

impl PreparedStep {
    pub fn padded(&self) -> (usize, usize) {
        self.pad.padded_size(self.width, self.height)
    }
}

/// Ramène une image à la résolution de travail (identité si déjà bonne).
fn resize_work(image: &RgbImage, work: (usize, usize)) -> RgbImage {
    let (w, h) = image.dimensions();
    if w as usize == work.0 && h as usize == work.1 {
        return image.clone();
    }
    imageops::resize(
        image,
        work.0 as u32,
        work.1 as u32,
        imageops::FilterType::Triangle,
    )
}

/// Prépare la frame et (éventuellement) son exemplaire.
///
/// `frame` est la frame source (le média la fournit désaturée) ; `reference`
/// est l'image-exemplaire **en couleur**, alignée sur la même géométrie ;
/// `padded` est la forme complétée attendue par les graphes.
pub fn prepare(
    frame: &RgbImage,
    reference: Option<&RgbImage>,
    padded: (usize, usize),
) -> PreparedStep {
    let (fw, fh) = frame.dimensions();
    let work = work_size(fw, fh, padded.0, padded.1);
    let frame = resize_work(frame, work);
    let (width, height) = work;
    let plane = width * height;

    let mut l = vec![0.0f32; plane];
    for (i, px) in frame.pixels().enumerate() {
        let (lightness, _, _) = rgb_to_lab(px[0], px[1], px[2]);
        l[i] = (lightness - 50.0) / 50.0;
    }
    let mut image = Vec::with_capacity(3 * plane);
    for _ in 0..3 {
        image.extend_from_slice(&l);
    }

    let exemplar = reference.map(|reference| {
        let reference = resize_work(reference, work);
        let mut ab = vec![0.0f32; 2 * plane];
        for (i, px) in reference.pixels().enumerate() {
            let (_, a, b) = rgb_to_lab(px[0], px[1], px[2]);
            ab[i] = a / 110.0;
            ab[plane + i] = b / 110.0;
        }
        ab
    });

    // Les graphes travaillent sur des dimensions multiples de 112 : on complète
    // par des zéros, marges réparties à parts égales (comme l'amont).
    let pad = pad_amounts(width, height);

    PreparedStep {
        image: pad_chw(&image, 3, width, height, pad),
        exemplar: exemplar.map(|ab| pad_chw(&ab, 2, width, height, pad)),
        width,
        height,
        pad,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    // Les littéraux sont les valeurs **float64** exactes de `skimage` : les
    // tronquer à la précision f32 effacerait la référence qu'ils documentent.
    #[allow(clippy::excessive_precision)]
    fn lab_matches_skimage_on_reference_values() {
        // Valeurs **calculées par `skimage.color.rgb2lab`** (0.25.2) : c'est la
        // référence de l'amont, pas une estimation. Noter que les gris parfaits
        // ne donnent pas exactement a = b = 0 (la matrice sRGB et le point blanc
        // ne se compensent qu'à ~3·10⁻³) — un détail que la conversion OpenCV de
        // `media` ne reproduit pas, d'où la conversion locale.
        let cases: [(u8, u8, u8, f32, f32, f32); 5] = [
            (0, 0, 0, 0.0, 0.0, 0.0),
            (255, 255, 255, 100.0, -0.002_455, 0.004_653),
            (128, 128, 128, 53.585_013, -0.001_473, 0.002_791),
            (255, 0, 0, 53.240_588, 80.092_308, 67.202_751),
            (0, 0, 255, 32.295_673, 79.185_591, -107.857_3),
        ];
        for (r, g, b, el, ea, eb) in cases {
            let (l, a, bb) = rgb_to_lab(r, g, b);
            assert!((l - el).abs() < 2e-3, "L({r},{g},{b}) = {l} vs {el}");
            assert!((a - ea).abs() < 5e-3, "a({r},{g},{b}) = {a} vs {ea}");
            assert!((bb - eb).abs() < 5e-3, "b({r},{g},{b}) = {bb} vs {eb}");
        }
    }

    #[test]
    fn padding_is_symmetric_and_zero_filled() {
        let frame = RgbImage::from_pixel(640, 360, Rgb([128, 128, 128]));
        let step = prepare(&frame, None, (672, 448));
        assert_eq!(step.padded(), (672, 448));
        assert_eq!(step.image.len(), 3 * 672 * 448);
        // 640×360 est déjà dans la fenêtre : aucune rééchantillonnage, on
        // retrouve exactement le pipeline amont (marges 16/16/44/44).
        assert_eq!((step.width, step.height), (640, 360));
        assert_eq!(step.pad, crate::geometry::pad_amounts(640, 360));
        // hors de la frame : zéro
        assert_eq!(step.image[0], 0.0);
        // dans la frame : (L(128) - 50) / 50
        let inside = step.pad.top * 672 + step.pad.left;
        assert!((step.image[inside] - (53.585_013 - 50.0) / 50.0).abs() < 5e-3);
    }

    #[test]
    fn the_work_size_preserves_the_aspect_ratio() {
        // 4:3 (le second extrait de référence) : le moteur travaillait en 16:9 et
        // étirait la source de 33 % — la résolution est maintenant dérivée.
        let (w, h) = work_size(320, 240, 672, 448);
        assert_eq!((w, h), (597, 448));
        assert_eq!(pad_amounts(w, h).padded_size(w, h), (672, 448));
        assert!((w as f64 / h as f64 - 4.0 / 3.0).abs() < 0.02, "rapport {w}/{h}");

        // Toute source déjà dans la fenêtre est laissée **native** : c'est ce qui
        // garde le portage comparable à la référence Python (640×360 → 672×448).
        assert_eq!(work_size(640, 360, 672, 448), (640, 360));
        assert_eq!(work_size(600, 400, 672, 448), (600, 400));

        // Un 16:9 plus grand que la fenêtre est ramené en 672×378.
        let (w, h) = work_size(1920, 1080, 672, 448);
        assert_eq!((w, h), (672, 378));
        assert_eq!(pad_amounts(w, h).padded_size(w, h), (672, 448));
        assert!((w as f64 / h as f64 - 16.0 / 9.0).abs() < 0.02);

        // Une source minuscule est bornée (formes statiques obligent).
        let (w, h) = work_size(64, 64, 672, 448);
        assert_eq!(pad_amounts(w, h).padded_size(w, h), (672, 448));
        assert!(w >= 1 && h >= 1);
    }

    #[test]
    // Idem : valeurs de référence `skimage` conservées telles quelles.
    #[allow(clippy::excessive_precision)]
    fn the_exemplar_is_optional_and_normalised() {
        let frame = RgbImage::from_pixel(640, 360, Rgb([0, 0, 0]));
        assert!(prepare(&frame, None, (672, 448)).exemplar.is_none());
        let reference = RgbImage::from_pixel(640, 360, Rgb([255, 0, 0]));
        let step = prepare(&frame, Some(&reference), (672, 448));
        let ab = step.exemplar.expect("exemplaire");
        assert_eq!(ab.len(), 2 * 672 * 448);
        let inside = step.pad.top * 672 + step.pad.left;
        assert!((ab[inside] - 80.092_308 / 110.0).abs() < 5e-4);
        assert!((ab[672 * 448 + inside] - 67.202_751 / 110.0).abs() < 5e-4);
        // les marges restent neutres (a = b = 0)
        assert_eq!(ab[0], 0.0);
    }

    #[test]
    fn unpad_is_the_inverse_of_pad() {
        let pad = pad_amounts(640, 360);
        let src: Vec<f32> = (0..2 * 640 * 360).map(|i| i as f32).collect();
        let padded = pad_chw(&src, 2, 640, 360, pad);
        assert_eq!(padded.len(), 2 * 672 * 448);
        let back = unpad_chw(&padded, 2, 640, 360, pad);
        assert_eq!(back, src);
    }

    #[test]
    fn padded_work_size_matches_the_exported_graphs() {
        for (sw, sh) in [(640u32, 360u32), (320, 240), (1920, 1080), (1280, 720)] {
            let (w, h) = work_size(sw, sh, 672, 448);
            assert_eq!(
                pad_amounts(w, h).padded_size(w, h),
                (672, 448),
                "source {sw}x{sh} -> travail {w}x{h}"
            );
            assert!((w as f64 / h as f64 - sw as f64 / sh as f64).abs() < 0.25);
        }
        assert_eq!(672 % PAD_MULTIPLE, 0);
    }
}
