//! Lecture de la mémoire (affinité + agrégation) — portée du code amont.
//!
//! Trois opérations, toutes en produits matriciels :
//!  1. `similarity` — score entre les clés mémorisées et la requête
//!     (`model/memory_util.get_similarity`), avec le terme de sélection ;
//!  2. `top_k_softmax` — normalisation sur les `k` meilleurs scores seulement
//!     (`do_softmax`) : les autres contributions sont nulles, ce qui borne le coût ;
//!  3. `readout` — moyenne pondérée des valeurs mémorisées (`readout`).
//!
//! Disposition mémoire : tous les plans `[N, HW]` sont stockés en lignes
//! (`idx = i * hw + j`), les tenseurs à canaux en `[C, N]` (`idx = c * n + i`).
//! Cette convention est celle des tenseurs ONNX (NCHW aplati) et évite toute
//! transposition à l'exécution.
//!
//! Vérifié par test contre un cas de référence **calculé par le code amont**
//! (`tests/fixtures/memory_readout.json`).

/// Scores de similarité entre mémoire et requête.
///
/// * `memory_keys` : `[ck, n]` — clés mémorisées
/// * `shrinkage`   : `[n]` — facteur par emplacement mémoire
/// * `query_keys`  : `[ck, hw]`
/// * `selection`   : `[ck, hw]` (facultatif) — terme de sélection de la requête
///
/// Retourne `[n, hw]`.
#[cfg(test)]
fn similarity_reference(
    memory_keys: &[f32],
    shrinkage: &[f32],
    query_keys: &[f32],
    selection: Option<&[f32]>,
    ck: usize,
    n: usize,
    hw: usize,
) -> Vec<f32> {
    assert_eq!(memory_keys.len(), ck * n);
    assert_eq!(query_keys.len(), ck * hw);
    let mut out = vec![0.0f32; n * hw];
    let scale = 1.0 / (ck as f32).sqrt();

    // Accès **contigus** : `memory_keys` est rangé canal-majeur (`c * n + i`) et
    // `query_keys` rangé requête-majeure (`c * hw + j`). La version naïve bouclait
    // `i → j → c` et lisait donc les deux tableaux avec un pas de `n` et de `hw` —
    // mesuré : 54 % du temps de rendu, à 0,44 GFLOP/s. On boucle `i → c → j` : un
    // scalaire de mémoire par `(i, c)`, puis des axpy **contigus** sur `j`.
    let mut a_sq = vec![0.0f32; hw];
    let mut two_ab = vec![0.0f32; hw];
    let mut b_sq = vec![0.0f32; hw];
    if let Some(selection) = selection {
        for c in 0..ck {
            let qrow = &query_keys[c * hw..(c + 1) * hw];
            let srow = &selection[c * hw..(c + 1) * hw];
            for j in 0..hw {
                let qk = qrow[j];
                b_sq[j] += srow[j] * qk * qk;
            }
        }
    }

    for i in 0..n {
        a_sq.fill(0.0);
        two_ab.fill(0.0);
        match selection {
            Some(selection) => {
                for c in 0..ck {
                    let mk = memory_keys[c * n + i];
                    let mk2 = mk * mk;
                    let qrow = &query_keys[c * hw..(c + 1) * hw];
                    let srow = &selection[c * hw..(c + 1) * hw];
                    for j in 0..hw {
                        let qe = srow[j];
                        a_sq[j] += mk2 * qe;
                        two_ab[j] += mk * qrow[j] * qe;
                    }
                }
            }
            None => {
                for c in 0..ck {
                    let mk = memory_keys[c * n + i];
                    let mk2 = mk * mk;
                    let qrow = &query_keys[c * hw..(c + 1) * hw];
                    for j in 0..hw {
                        a_sq[j] += mk2;
                        two_ab[j] += mk * qrow[j];
                    }
                }
            }
        }
        let row = &mut out[i * hw..(i + 1) * hw];
        let weight = shrinkage[i] * scale;
        for j in 0..hw {
            row[j] = (-a_sq[j] + 2.0 * two_ab[j] - b_sq[j]) * weight;
        }
    }
    out
}

/// Similarité mémoire ↔ requête, `[n, hw]`, en **produits matriciels bloqués**.
///
/// Les deux termes se factorisent :
/// * `two_ab = Σ_c mk·qk·qe` — une requête **pondérée** `qw = qk·qe` rend le terme
///   linéaire : `two_ab = MKᵀ·QW` ;
/// * `a_sq = Σ_c mk²·qe = (MK²)ᵀ·QE`.
///
/// Ce sont donc deux produits matriciels écrits en **mise à jour rang-1 par canal**
/// (le canal est la dimension de contraction) : chaque `(i, c)` charge un scalaire,
/// chaque `(c, j)` charge un vecteur contigu, et une tuile `TI × TJ` reste en
/// registres. L'implémentation scalaire faisait 885 MFLOP/pas à 0,5 GFLOP/s ; c'est
/// le premier poste de temps du moteur (54 %).
pub fn similarity(
    memory_keys: &[f32],
    shrinkage: &[f32],
    query_keys: &[f32],
    selection: Option<&[f32]>,
    ck: usize,
    n: usize,
    hw: usize,
) -> Vec<f32> {
    assert_eq!(memory_keys.len(), ck * n);
    assert_eq!(query_keys.len(), ck * hw);
    let mut out = vec![0.0f32; n * hw];
    let scale = 1.0 / (ck as f32).sqrt();

    let Some(selection) = selection else {
        // Chemin sans pondération : même structure, une seule accumulation.
        return similarity_unweighted(memory_keys, shrinkage, query_keys, ck, n, hw);
    };
    assert_eq!(selection.len(), ck * hw);

    // b_sq ne dépend que de (c, j).
    let mut b_sq = vec![0.0f32; hw];
    for c in 0..ck {
        let qrow = &query_keys[c * hw..(c + 1) * hw];
        let srow = &selection[c * hw..(c + 1) * hw];
        for j in 0..hw {
            let qk = qrow[j];
            b_sq[j] += srow[j] * qk * qk;
        }
    }
    // Requête pondérée, calculée une fois : `qw[c][j] = qk·qe`.
    let qw: Vec<f32> = query_keys
        .iter()
        .zip(selection)
        .map(|(qk, qe)| qk * qe)
        .collect();

    const TI: usize = 32;
    const TJ: usize = 64;
    let mut i0 = 0;
    while i0 < n {
        let ti = TI.min(n - i0);
        let mut j0 = 0;
        while j0 < hw {
            let tj = TJ.min(hw - j0);
            let mut acc_a = [[0.0f32; TJ]; TI];
            let mut acc_b = [[0.0f32; TJ]; TI];
            for c in 0..ck {
                let mk = &memory_keys[c * n + i0..c * n + i0 + ti];
                let qe = &selection[c * hw + j0..c * hw + j0 + tj];
                let qr = &qw[c * hw + j0..c * hw + j0 + tj];
                for ii in 0..ti {
                    let m = mk[ii];
                    let m2 = m * m;
                    let row_a = &mut acc_a[ii];
                    let row_b = &mut acc_b[ii];
                    for jj in 0..tj {
                        row_a[jj] += m2 * qe[jj];
                        row_b[jj] += m * qr[jj];
                    }
                }
            }
            for ii in 0..ti {
                let weight = shrinkage[i0 + ii] * scale;
                let row = &mut out[(i0 + ii) * hw + j0..(i0 + ii) * hw + j0 + tj];
                for jj in 0..tj {
                    row[jj] = (-acc_a[ii][jj] + 2.0 * acc_b[ii][jj] - b_sq[j0 + jj]) * weight;
                }
            }
            j0 += TJ;
        }
        i0 += TI;
    }
    out
}

/// Même calcul sans pondération de requête (`selection` absente).
fn similarity_unweighted(
    memory_keys: &[f32],
    shrinkage: &[f32],
    query_keys: &[f32],
    ck: usize,
    n: usize,
    hw: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n * hw];
    let scale = 1.0 / (ck as f32).sqrt();
    let mut a_sq = vec![0.0f32; hw];
    let mut two_ab = vec![0.0f32; hw];
    for i in 0..n {
        a_sq.fill(0.0);
        two_ab.fill(0.0);
        for c in 0..ck {
            let mk = memory_keys[c * n + i];
            let mk2 = mk * mk;
            let qrow = &query_keys[c * hw..(c + 1) * hw];
            for j in 0..hw {
                a_sq[j] += mk2;
                two_ab[j] += mk * qrow[j];
            }
        }
        let weight = shrinkage[i] * scale;
        let row = &mut out[i * hw..(i + 1) * hw];
        for j in 0..hw {
            row[j] = (-a_sq[j] + 2.0 * two_ab[j]) * weight;
        }
    }
    out
}

/// `top_k` meilleurs scores par colonne, en forme **creuse** : indices et poids.
///
/// La forme dense (`n × hw`) faisait 300 Mo par pas dont 99,7 % de zéros, et le
/// `readout` multipliait par ces zéros : 14,2 GFLOP mesurés, 69 % du temps du
/// moteur. Garder la sélection creuse supprime et le remplissage, et le calcul.
#[derive(Debug, Clone, PartialEq)]
pub struct TopK {
    /// Indices mémoire, `[k, hw]`.
    pub indices: Vec<u32>,
    /// Poids softmax correspondants, `[k, hw]`.
    pub weights: Vec<f32>,
    pub k: usize,
    pub hw: usize,
}

/// Sélectionne les `top_k` meilleurs scores de chaque colonne et les normalise.
///
/// Sélection **partielle** (`select_nth_unstable`) plutôt qu'un tri complet : le
/// tri de 11 760 éléments répété 1 176 fois coûtait 252 ms par pas.
pub fn top_k_softmax(similarity: &[f32], n: usize, hw: usize, top_k: usize) -> TopK {
    assert_eq!(similarity.len(), n * hw);
    let k = top_k.min(n).max(1);
    let mut indices = vec![0u32; k * hw];
    let mut weights = vec![0.0f32; k * hw];
    let mut order: Vec<u32> = (0..n as u32).collect();

    for j in 0..hw {
        // `select_nth_unstable_by` place les k plus grands en tête, sans trier le
        // reste : O(n) au lieu de O(n log n).
        let column = |i: &u32| similarity[*i as usize * hw + j];
        // `select_nth_unstable_by` garantit que `order[..k]` contient les k plus
        // grands — le k-ième *inclus*. Ne trier que le `head` rendu en oublierait
        // un (bug attrapé par la fixture amont).
        order.select_nth_unstable_by(k - 1, |a, b| {
            column(b).partial_cmp(&column(a)).unwrap_or(std::cmp::Ordering::Equal)
        });
        order[..k].sort_unstable_by(|a, b| {
            column(b).partial_cmp(&column(a)).unwrap_or(std::cmp::Ordering::Equal)
        });

        let max = column(&order[0]);
        let mut sum = 0.0f32;
        for (t, &i) in order[..k].iter().enumerate() {
            let e = (column(&i) - max).exp();
            indices[t * hw + j] = i;
            weights[t * hw + j] = e;
            sum += e;
        }
        if sum > 0.0 {
            for t in 0..k {
                weights[t * hw + j] /= sum;
            }
        }
    }
    TopK { indices, weights, k, hw }
}

/// Moyenne pondérée des valeurs mémorisées : `[cv, n]` × `[n, hw]` → `[cv, hw]`.
///
/// Produit matriciel, et **de loin l'étage le plus coûteux** du moteur : à la
/// taille réelle du clip de référence (`cv=512`, `hw=1176`, `n` jusqu'à 11 760),
/// il pèse ~4,4 s par pas. La boucle est donc écrite en `axpy` par ligne — les
/// deux accès sont contigus — plutôt qu'en accumulation scalaire avec un accès
/// de pas `hw` (mesuré ~4× plus lent, cf. `memory_cost_profile`).
///
/// L'ordre des sommes sur `i` est **préservé** : les deux formes donnent le même
/// résultat au bit près.
pub fn readout(selection: &TopK, values: &[f32], cv: usize, n: usize, hw: usize) -> Vec<f32> {
    assert_eq!(values.len(), cv * n);
    assert_eq!(selection.hw, hw);
    let k = selection.k;
    let mut out = vec![0.0f32; cv * hw];

    // Boucle **canal-majeur** : `values` est rangé `[cv, n]`, donc la ligne d'un
    // canal est contiguë (47 Ko aux formes réelles — elle tient en L2, les accès
    // indirects par les indices restent donc bon marché). La somme sur `t` ne
    // parcourt que les `k` contributions non nulles, au lieu des `n` colonnes.
    for (c, row) in out.chunks_exact_mut(hw).enumerate() {
        let values_row = &values[c * n..(c + 1) * n];
        for (j, acc) in row.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for t in 0..k {
                let index = selection.indices[t * hw + j] as usize;
                sum += selection.weights[t * hw + j] * values_row[index];
            }
            *acc = sum;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// Où passe le temps ? Aux formes réelles du moteur (mémoire pleine :
    /// 10 frames × 1176 colonnes, 64 canaux, 1176 requêtes), décomposé phase par
    /// phase. `cargo test -p colorvid-quality-colormnet --lib -- --ignored --nocapture`.
    #[test]
    #[ignore = "mesure de performance : à lancer explicitement"]
    fn profile_similarity_phases() {
        use std::time::Instant;
        let (ck, hw, frames) = (64usize, 28 * 42, 10usize);
        let n = frames * hw;
        let memory: Vec<f32> = (0..ck * n).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
        let query: Vec<f32> = (0..ck * hw).map(|i| ((i * 53 % 97) as f32 - 48.0) / 48.0).collect();
        let selection: Vec<f32> = (0..ck * hw).map(|i| ((i * 29 % 89) as f32) / 89.0).collect();
        let shrinkage: Vec<f32> = (0..n).map(|i| 0.5 + (i % 7) as f32 / 10.0).collect();
        let values: Vec<f32> = (0..512 * n).map(|i| ((i * 17 % 71) as f32 - 35.0) / 35.0).collect();

        let t = Instant::now();
        let sim = similarity(&memory, &shrinkage, &query, Some(&selection), ck, n, hw);
        let t_sim = t.elapsed();

        let t = Instant::now();
        let weights = top_k_softmax(&sim, n, hw, 30);
        let t_topk = t.elapsed();

        let t = Instant::now();
        let out = readout(&weights, &values, 512, n, hw);
        let t_read = t.elapsed();

        println!("formes : n={n} hw={hw} ck={ck}");
        println!("  similarity   {:8.1} ms", t_sim.as_secs_f64() * 1e3);
        println!("  top_k_softmax{:8.1} ms", t_topk.as_secs_f64() * 1e3);
        println!("  readout      {:8.1} ms", t_read.as_secs_f64() * 1e3);
        println!("  total        {:8.1} ms", (t_sim + t_topk + t_read).as_secs_f64() * 1e3);
        assert_eq!(out.len(), 512 * hw);
    }

    /// La version bloquée doit rendre **le même résultat** que la version scalaire
    /// qu'elle remplace. Celle-ci est conservée comme oracle : sans ce test, une
    /// optimisation de 885 MFLOP/pas serait invérifiable.
    #[test]
    fn blocked_similarity_matches_the_scalar_reference() {
        // tailles qui traversent les tuiles (TI = 32, TJ = 64) sans être énormes
        for (ck, n, hw) in [(16, 70, 70), (64, 100, 200), (8, 5, 3), (32, 128, 64)] {
            let memory: Vec<f32> = (0..ck * n).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
            let query: Vec<f32> = (0..ck * hw).map(|i| ((i * 53 % 97) as f32 - 48.0) / 48.0).collect();
            let selection: Vec<f32> = (0..ck * hw).map(|i| ((i * 29 % 89) as f32) / 89.0).collect();
            let shrinkage: Vec<f32> = (0..n).map(|i| 0.5 + (i % 7) as f32 / 10.0).collect();

            for choice in [None, Some(selection.as_slice())] {
                let got = similarity(&memory, &shrinkage, &query, choice, ck, n, hw);
                let want = similarity_reference(&memory, &shrinkage, &query, choice, ck, n, hw);
                assert_eq!(got.len(), want.len());
                let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
                let worst = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    worst <= 1e-4 * scale,
                    "ck={ck} n={n} hw={hw} pondéré={}: écart {worst:.3e} pour une amplitude {scale:.3e}",
                    choice.is_some()
                );
            }
        }
    }

    /// Cas de référence calculé par `model/memory_util.py` (code amont) :
    /// c'est la vérité de test, pas une réimplémentation de ma part.
    fn fixture() -> Value {
        let raw = include_str!("../tests/fixtures/memory_readout.json");
        serde_json::from_str(raw).expect("fixture JSON")
    }

    fn f32s(v: &Value, key: &str) -> Vec<f32> {
        v[key]
            .as_array()
            .expect(key)
            .iter()
            .map(|x| x.as_f64().expect("nombre") as f32)
            .collect()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn matches_the_upstream_reference_case() {
        let f = fixture();
        let ck = f["ck"].as_u64().unwrap() as usize;
        let top_k = f["top_k"].as_u64().unwrap() as usize;
        let mk = f32s(&f, "mk");
        let ms = f32s(&f, "ms");
        let qk = f32s(&f, "qk");
        let qe = f32s(&f, "qe");
        let mv = f32s(&f, "mv");
        let n = mk.len() / ck;
        let hw = qk.len() / ck;
        let cv = mv.len() / n;

        let sim = similarity(&mk, &ms, &qk, Some(&qe), ck, n, hw);
        assert!(
            max_abs_diff(&sim, &f32s(&f, "similarity")) < 1e-5,
            "similarité divergente"
        );

        let aff = top_k_softmax(&sim, n, hw, top_k);
        // La fixture amont est la matrice **dense** : on la reconstruit depuis la
        // forme creuse — c'est justement la garantie que le passage au creux ne
        // change pas le résultat.
        let mut dense = vec![0.0f32; n * hw];
        for t in 0..aff.k {
            for j in 0..hw {
                dense[aff.indices[t * hw + j] as usize * hw + j] = aff.weights[t * hw + j];
            }
        }
        assert!(
            max_abs_diff(&dense, &f32s(&f, "affinity")) < 1e-6,
            "affinité divergente"
        );

        let mem = readout(&aff, &mv, cv, n, hw);
        assert!(
            max_abs_diff(&mem, &f32s(&f, "readout")) < 1e-5,
            "lecture mémoire divergente"
        );
    }

    #[test]
    fn top_k_softmax_normalises_and_keeps_only_k() {
        // n = 6, top_k = 2 : deux contributions par colonne, et les bonnes.
        let sim = vec![0.1, 5.0, 0.2, 4.0, 0.3, 0.4];
        let aff = top_k_softmax(&sim, 6, 1, 2);
        assert_eq!(aff.k, 2);
        assert_eq!(aff.indices, vec![1, 3]); // les deux plus grands scores
        assert!((aff.weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn readout_is_a_weighted_average() {
        // une seule mémoire active : la sortie vaut la valeur correspondante.
        let selection = TopK { indices: vec![0], weights: vec![1.0], k: 1, hw: 1 };
        let values = vec![3.0f32, 7.0, -1.0, 5.0]; // cv = 2, n = 2
        let out = readout(&selection, &values, 2, 2, 1);
        assert_eq!(out, vec![3.0, -1.0]);
    }
}

#[cfg(test)]
mod cost {
    use super::*;
    use std::time::Instant;

    /// Profil de coût des opérations mémoire **en Rust pur**, à la taille réelle
    /// du plan le plus long du clip de référence (10 frames en mémoire).
    ///
    /// Sert au jalon C7 : la lecture mémoire est le seul étage dont le coût
    /// croît avec la durée du plan, et une implémentation naïve en boucles
    /// triple est nettement plus lente qu'un BLAS.
    ///
    /// `cargo test -p colorvid-quality-colormnet --release memory::cost -- --ignored --nocapture`
    #[test]
    #[ignore = "profil de coût (quelques secondes)"]
    fn memory_cost_profile() {
        let (ck, cv, hw) = (64usize, 512usize, 28 * 42);
        for frames in [1usize, 5, 10] {
            let n = frames * hw;
            let mk = vec![0.01f32; ck * n];
            let ms = vec![0.9f32; n];
            let qk = vec![0.01f32; ck * hw];
            let qe = vec![0.5f32; ck * hw];
            let mv = vec![0.01f32; cv * n];

            let t = Instant::now();
            let sim = similarity(&mk, &ms, &qk, Some(&qe), ck, n, hw);
            let t_sim = t.elapsed().as_secs_f64();

            let t = Instant::now();
            let aff = top_k_softmax(&sim, n, hw, 30);
            let t_aff = t.elapsed().as_secs_f64();

            let t = Instant::now();
            let _ = readout(&aff, &mv, cv, n, hw);
            let t_read = t.elapsed().as_secs_f64();

            println!(
                "{frames:2} frames en mémoire (n={n:6}) : similarité {t_sim:6.3} s | \
                 top-k {t_aff:6.3} s | agrégation {t_read:6.3} s | total {:6.3} s",
                t_sim + t_aff + t_read
            );
        }
    }
}
