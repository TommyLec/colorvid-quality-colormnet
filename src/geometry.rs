//! Géométrie d'entrée de ColorMNet.
//!
//! Le réseau travaille sur des dimensions multiples de **112** (patch DINOv2 de
//! 14 × 8) : la frame est complétée par des zéros, puis le résultat est recadré.
//! Les marges sont réparties à parts égales, le pixel excédentaire allant à
//! droite et en bas — sémantique reprise de `util/tensor_util.pad_divide_by`.

/// Multiple d'alignement attendu par le réseau.
pub const PAD_MULTIPLE: usize = 112;

/// Marges appliquées autour de la frame (en pixels).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PadAmounts {
    pub left: usize,
    pub right: usize,
    pub top: usize,
    pub bottom: usize,
}

impl PadAmounts {
    /// Vrai si aucune marge n'est nécessaire (dimensions déjà alignées).
    pub fn is_identity(&self) -> bool {
        self.left == 0 && self.right == 0 && self.top == 0 && self.bottom == 0
    }

    /// Dimensions après complétion.
    pub fn padded_size(&self, width: usize, height: usize) -> (usize, usize) {
        (
            width + self.left + self.right,
            height + self.top + self.bottom,
        )
    }
}

/// Calcule les marges pour amener `width`/`height` au multiple de 112 supérieur.
pub fn pad_amounts(width: usize, height: usize) -> PadAmounts {
    let align = |v: usize| {
        let rem = v % PAD_MULTIPLE;
        if rem == 0 {
            (v, 0, 0)
        } else {
            let total = PAD_MULTIPLE - rem;
            (v + total, total / 2, total - total / 2)
        }
    };
    let (_, lw, uw) = align(width);
    let (_, lh, uh) = align(height);
    PadAmounts {
        left: lw,
        right: uw,
        top: lh,
        bottom: uh,
    }
}

/// Dimensions effectivement traitées par le réseau.
pub fn padded_size(width: usize, height: usize) -> (usize, usize) {
    pad_amounts(width, height).padded_size(width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_reference_padding_for_the_test_clip() {
        // 640x360 -> 672x448 (marges 16/16/44/44) : valeurs vérifiées sur
        // l'implémentation PyTorch (formes réellement capturées : 1x3x448x672).
        let pad = pad_amounts(640, 360);
        assert_eq!(
            pad,
            PadAmounts { left: 16, right: 16, top: 44, bottom: 44 }
        );
        assert_eq!(padded_size(640, 360), (672, 448));
        assert_eq!(pad.padded_size(640, 360), (672, 448));
    }

    #[test]
    fn aligned_sizes_are_untouched() {
        assert!(pad_amounts(224, 224).is_identity());
        assert!(pad_amounts(672, 448).is_identity());
        assert_eq!(padded_size(112, 112), (112, 112));
    }

    #[test]
    fn odd_remainder_goes_to_the_right_and_bottom() {
        // h = 113 : reste 1 -> total 111, moitié basse 55, haute 56.
        let pad = pad_amounts(112, 113);
        assert_eq!(pad.left + pad.right, 0);
        assert_eq!((pad.top, pad.bottom), (55, 56));
        assert!(pad.padded_size(112, 113).1.is_multiple_of(PAD_MULTIPLE));
    }

    #[test]
    fn padded_sizes_are_always_multiples() {
        for w in [1usize, 111, 112, 113, 640, 1280] {
            for h in [1usize, 111, 112, 113, 360, 720] {
                let (pw, ph) = padded_size(w, h);
                assert_eq!(pw % PAD_MULTIPLE, 0, "largeur {w}");
                assert_eq!(ph % PAD_MULTIPLE, 0, "hauteur {h}");
                assert!(pw >= w && pw - w < PAD_MULTIPLE);
                assert!(ph >= h && ph - h < PAD_MULTIPLE);
            }
        }
    }
}
