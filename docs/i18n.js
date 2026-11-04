// French / English switch. English lives in index.html (the source); French is below.
// Static text: elements with data-i18n (innerHTML), data-i18n-alt (img alt), data-i18n-aria (aria-label).
// Dynamic text built in app.js goes through I18N.t(key, ...args).
(function () {
  const FR = {
    kicker: "Étude computationnelle · transcriptomes publics de cerveaux post-mortem",
    title: "Les cerveaux ayant subi un traumatisme crânien ressemblent-ils à des cerveaux atteints d'Alzheimer ?",
    lede: "Un modèle apprend la signature d'expression génique de la maladie d'Alzheimer dans trois cohortes, puis note " +
      "376 échantillons de cerveau de 106 donneurs de l'étude Allen <em>Aging, Dementia and TBI</em>, qu'il n'a jamais vus.",
    nav_code: "Code sur GitHub", nav_explore: "Explorer les scores", nav_limits: "Limites",
    answer_h: "Réponse courte : un peu, et spécifiquement dans la direction d'Alzheimer.",
    answer_p: "Les donneurs ayant eu un traumatisme crânien (TC) avec perte de connaissance obtiennent un score légèrement plus " +
      "élevé sur la signature d'Alzheimer que leurs témoins appariés sans TC. Ce décalage est spécifique à la direction " +
      "d'Alzheimer, mais faible, et aucun gène ne le porte à lui seul. Un réseau de neurones sur graphe n'a <strong>pas</strong> " +
      "fait mieux qu'un modèle linéaire.",
    note: "Tous les résultats sont des associations statistiques dans du tissu post-mortem. Ils ne montrent pas que le " +
      "traumatisme crânien cause la maladie d'Alzheimer et n'ont aucun usage clinique.",
    explorer_h: "Explorateur de scores",
    explorer_p: "Chaque point est un échantillon de cerveau. Les scores sont en écarts-types sur la cohorte Allen ; plus le " +
      "score est élevé, plus l'expression ressemble à celle d'Alzheimer. Cliquez sur un point ou choisissez un échantillon dans la liste.",
    lbl_model: "Modèle", lbl_region: "Région", lbl_sample: "Échantillon", lbl_gene: "Gène",
    m_lr_l2: "Régression logistique L2 (meilleur)", m_gat: "GAT sur le graphe STRING", m_lr_l1: "Régression logistique L1",
    m_mlp: "MLP", m_rf: "Forêt aléatoire",
    r_all: "Toutes les régions", r_TCx: "Cortex temporal", r_PCx: "Cortex pariétal", r_HIP: "Hippocampe", r_FWM: "Substance blanche",
    aria_swarm: "Scores de ressemblance à Alzheimer des échantillons avec et sans traumatisme crânien",
    genes_drive_h: "Gènes qui déterminent le score de cet échantillon",
    genes_drive_p: "Contribution en régression logistique (coefficient × expression standardisée). " +
      "<span class=\"sw pos\"></span> pousse vers Alzheimer, <span class=\"sw neg\"></span> s'en éloigne.",
    ego_h: "Voisinages de gènes",
    ego_p: "Les 102 gènes dont les poids sont stables entre les cinq modèles finaux de régression logistique, avec leurs " +
      "partenaires STRING à haute confiance. La couleur indique comment le gène varie dans Alzheimer (z de méta-analyse) ; " +
      "la taille indique son poids dans le score.",
    aria_ego: "Voisinage STRING du gène sélectionné",
    find_h: "Ce que nous avons trouvé",
    f1: "<strong>Une signature d'Alzheimer robuste.</strong> <span data-k=\"adsig\"></span> gènes varient de façon cohérente dans " +
      "trois cohortes d'entraînement ; <span data-k=\"repl\"></span> d'entre eux varient dans le même sens dans une quatrième " +
      "cohorte indépendante. Les gènes neuronaux, synaptiques et mitochondriaux diminuent ; les gènes vasculaires, TGF-β et " +
      "de l'immunité innée augmentent.",
    f2: "<strong>Aucun gène propre au TC.</strong> Dans aucune région, aucun gène n'est significativement associé au traumatisme " +
      "crânien. Les tests de chevauchement gène par gène avec Alzheimer semblent très significatifs dans l'hippocampe, mais " +
      "s'effondrent dès qu'on permute les donneurs plutôt que les gènes (toutes régions : p = <span data-k=\"globp\"></span>) : " +
      "la corrélation entre gènes rend les tests naïfs trop optimistes.",
    f3: "<strong>Un décalage modeste et spécifique à Alzheimer, par échantillon.</strong> Les donneurs avec TC ont un score " +
      "supérieur de <span data-k=\"pair\"></span> ÉT à celui de leurs témoins appariés (test exact par inversion de signe : " +
      "p = <span data-k=\"sfp\"></span> ; modèle mixte ajusté sur l'âge, le sexe, la démence, la qualité de l'ARN et la région : " +
      "<span data-k=\"beta\"></span> ÉT, p = <span data-k=\"mp\"></span>), avec un effet dose. Ce décalage dépasse celui de " +
      "<span data-k=\"rand\"></span> % des signatures géniques aléatoires. Il ne résiste pas à la correction pour les cinq " +
      "modèles et s'affaiblit chez les seuls donneurs non déments.",
    f4: "<strong>Le graphe n'a pas aidé.</strong> La régression logistique atteint une AUROC de <span data-k=\"lr\"></span>, le GAT " +
      "sur le réseau STRING <span data-k=\"gat\"></span>. Le même GAT sur un graphe aléatoire à degrés préservés obtient " +
      "<span data-k=\"gatr\"></span>, et sans arêtes <span data-k=\"gatn\"></span>. Sur une cohorte inédite, tous les modèles " +
      "chutent à environ 0,75.",
    f5: "<strong>Des facteurs diffus.</strong> Aucun gène, aucune voie et aucun ensemble de type cellulaire ne porte à lui seul le " +
      "décalage lié au TC. Les explications du GAT étaient instables d'un modèle à l'autre et son attention suivait simplement " +
      "le degré des nœuds ; l'interprétation repose donc sur le modèle linéaire.",
    fig_h: "Figures",
    fig_note: "Les légendes à l'intérieur des figures sont en anglais.",
    a1: "Scores de ressemblance à Alzheimer par région, avec et sans TC, et paires appariées",
    c1: "Scores par région et les 48 paires appariées TC / sans TC.",
    a2: "Décalage observé lié au TC face à 2 000 signatures géniques aléatoires",
    c2: "Le décalage lié au TC face à 2 000 signatures géniques aléatoires.",
    a3: "AUROC des variantes du GAT et des modèles de référence",
    c3: "Variantes du GAT face aux modèles de référence, validation croisée et cohorte laissée de côté.",
    a4: "p-valeurs hypergéométriques naïves face aux p-valeurs par permutation des paires appariées",
    c4: "Les tests gène par gène surestiment le chevauchement TC–Alzheimer.",
    a5: "Statistiques TC face aux z d'Alzheimer par région, avec distributions nulles par permutation",
    c5: "Statistiques géniques TC face à Alzheimer par région, avec distributions nulles par permutation.",
    a6: "Graphique volcan de la méta-analyse Alzheimer",
    c6: "La signature d'Alzheimer (méta-analyse de trois cohortes).",
    a7: "ACP avant correction des effets de lot", c7: "Avant correction des effets de lot : les échantillons se regroupent par jeu de données.",
    a8: "ACP après correction des effets de lot", c8: "Après correction : les jeux de données se mélangent et un axe Alzheimer apparaît.",
    a9: "Gènes stables du score sur le réseau STRING", c9: "Gènes stables du score sur le réseau STRING.",
    method_h: "Données et méthode",
    th_role: "Rôle", th_ds: "Jeu de données", th_tissue: "Tissu", th_donors: "Donneurs",
    role_train: "Entraînement", role_ext: "Test externe", role_tbi: "Cohorte TC",
    t1: "Cortex temporal", t2: "Temporal, entorhinal, frontal", t3: "Hippocampe et 3 cortex", t4: "Cortex préfrontal",
    t5: "4 régions, RNA-seq",
    s1: "Ramener quatre plateformes aux mêmes gènes ; corriger les effets de lot avec ComBat sans utiliser les étiquettes.",
    s2: "Biologie classique : méta-analyse Alzheimer, DESeq2 pour le TC par région, tests de chevauchement par permutation.",
    s3: "Graphe de gènes : arêtes STRING v12 à haute confiance sur 4 000 gènes sélectionnés.",
    s4: "Modèles : GAT avec lecture par gène face à la régression logistique, la forêt aléatoire et le MLP, validés par donneur.",
    s5: "Transfert : noter les échantillons Allen ; tester le TC par modèles mixtes, paires appariées et signatures aléatoires.",
    limits_h: "Limites",
    l1: "Petite cohorte TC : 106 donneurs et 48 paires appariées complètes ; des effets de 0,3 ÉT sont proches du seuil de détection.",
    l2: "Le TC est autodéclaré, avec perte de connaissance, souvent des décennies avant le décès ; pas de données de gravité.",
    l3: "Tissu post-mortem global : on ne peut pas séparer les changements de composition cellulaire de la régulation dans chaque cellule.",
    l4: "Quatre plateformes et une correction des effets de lot ; la performance tombe à environ 0,75 d'AUROC sur une cohorte inédite.",
    l5: "Plusieurs modèles et régions ont été testés ; l'effet principal ne résiste pas à la correction de Bonferroni.",
    l6: "Associations uniquement. Aucune affirmation causale ou clinique.",
    foot: "Données publiques : GEO et l'étude Allen Institute Aging, Dementia and TBI. STRING v12. Ensembles de gènes Enrichr. " +
      "Pipeline complet, rapports et étapes de reproduction dans le <a href=\"https://github.com/Lawl1et-1/tbi-ad-graphml\">dépôt</a>.",
  };

  // Dynamic strings used by app.js. {0}, {1}... are filled in order.
  const UI = {
    en: {
      card_pair: "TBI vs matched non-TBI donor, AD-likeness score (logistic regression)",
      card_sf: "exact matched-pair sign-flip test (48 pairs)",
      card_rand: "of random gene signatures show a smaller TBI shift",
      card_auc: "AUROC: GAT on STRING vs logistic regression",
      axis: "AD-likeness score (SD) → more Alzheimer-like", axis_short: "Score (SD) → more Alzheimer-like",
      tbi: "TBI", notbi: "no TBI",
      cap: "Vertical bars are group means.", cap_row: "{0}: TBI {1} (n={2}) vs no TBI {3} (n={4})",
      pct: "{0} percentile in {1}", sd: "SD",
      donor: "Donor", region: "Region", tbi_loc: "TBI with LOC", dx: "Clinical diagnosis", braak: "Braak stage", agesex: "Age / sex",
      yes: "yes", no: "no", dx_AD: "AD", dx_control: "control", dx_other_dementia: "other dementia",
      ego_cap: "{0}: Alzheimer meta z {1}, score weight {2} ({3}). {4} STRING partners among the 4,000 model genes; showing {5} with the largest weight.",
      toward: "pushes toward Alzheimer's", away: "pushes away from Alzheimer's",
      ego_none: "{0} has no high-confidence STRING partner among the 4,000 model genes.",
      dot_title: "{0} · {1} · {2} · {3} SD", load_err: "Could not load data: ",
    },
    fr: {
      card_pair: "Donneur avec TC vs témoin apparié sans TC, score de ressemblance à Alzheimer (régression logistique)",
      card_sf: "test exact par inversion de signe sur les paires (48 paires)",
      card_rand: "des signatures géniques aléatoires montrent un décalage plus faible",
      card_auc: "AUROC : GAT sur STRING vs régression logistique",
      axis: "Score de ressemblance à Alzheimer (ÉT) → plus proche d'Alzheimer", axis_short: "Score (ÉT) → plus proche d'Alzheimer",
      tbi: "TC", notbi: "sans TC",
      cap: "Les barres verticales sont les moyennes des groupes.", cap_row: "{0} : TC {1} (n={2}) vs sans TC {3} (n={4})",
      pct: "{0} centile ({1})", sd: "ÉT",
      donor: "Donneur", region: "Région", tbi_loc: "TC avec perte de connaissance", dx: "Diagnostic clinique",
      braak: "Stade de Braak", agesex: "Âge / sexe",
      yes: "oui", no: "non", dx_AD: "Alzheimer", dx_control: "témoin", dx_other_dementia: "autre démence",
      ego_cap: "{0} : z de méta-analyse Alzheimer {1}, poids dans le score {2} ({3}). {4} partenaires STRING parmi les 4 000 gènes du modèle ; {5} affichés, ceux de plus grand poids.",
      toward: "pousse vers Alzheimer", away: "éloigne d'Alzheimer",
      ego_none: "{0} n'a aucun partenaire STRING à haute confiance parmi les 4 000 gènes du modèle.",
      dot_title: "{0} · {1} · {2} · {3} ÉT", load_err: "Impossible de charger les données : ",
    },
  };

  const EN = {};
  let lang = "en";
  const listeners = [];

  function capture() {
    document.querySelectorAll("[data-i18n]").forEach((e) => { if (!(e.dataset.i18n in EN)) EN[e.dataset.i18n] = e.innerHTML; });
    document.querySelectorAll("[data-i18n-alt]").forEach((e) => { EN[e.dataset.i18nAlt] = e.getAttribute("alt"); });
    document.querySelectorAll("[data-i18n-aria]").forEach((e) => { EN[e.dataset.i18nAria] = e.getAttribute("aria-label"); });
    EN.__title = document.title;
  }

  function apply(next) {
    lang = next === "fr" ? "fr" : "en";
    const dict = lang === "fr" ? FR : EN;
    document.documentElement.lang = lang;
    document.querySelectorAll("[data-i18n]").forEach((e) => { const v = dict[e.dataset.i18n]; if (v !== undefined) e.innerHTML = v; });
    document.querySelectorAll("[data-i18n-alt]").forEach((e) => { const v = dict[e.dataset.i18nAlt]; if (v) e.setAttribute("alt", v); });
    document.querySelectorAll("[data-i18n-aria]").forEach((e) => { const v = dict[e.dataset.i18nAria]; if (v) e.setAttribute("aria-label", v); });
    document.title = lang === "fr" ? "Signatures TC et Alzheimer" : EN.__title;
    document.querySelectorAll(".lang button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === lang)));
    try { localStorage.setItem("lang", lang); } catch (_) { /* private mode: keep the choice for this visit only */ }
    listeners.forEach((f) => f(lang));
  }

  function t(key, ...args) {
    const s = (UI[lang] && UI[lang][key]) ?? UI.en[key] ?? key;
    return s.replace(/\{(\d+)\}/g, (_, i) => (args[i] ?? ""));
  }

  function initial() {
    try { const s = localStorage.getItem("lang"); if (s === "fr" || s === "en") return s; } catch (_) { /* ignore */ }
    return (navigator.language || "en").toLowerCase().startsWith("fr") ? "fr" : "en";
  }

  window.I18N = {
    get lang() { return lang; },
    t, apply, onChange: (f) => listeners.push(f),
    init() {
      capture();
      document.querySelectorAll(".lang button").forEach((b) => b.addEventListener("click", () => apply(b.dataset.lang)));
      apply(initial());
    },
  };
})();
