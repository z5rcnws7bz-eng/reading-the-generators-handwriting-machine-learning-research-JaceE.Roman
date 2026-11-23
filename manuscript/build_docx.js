// ESWA (Elsevier, Expert Systems with Applications) manuscript layout:
// A4, 1-inch margins, Times New Roman 12 pt, double spacing, continuous line numbers,
// title page -> highlights -> abstract -> keywords -> body -> declarations -> APA references -> appendix.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, BorderStyle,
  AlignmentType, HeadingLevel, Footer, PageNumber, LevelFormat, LineNumberRestartFormat, PageBreak,
  ShadingType, TabStopType, VerticalAlign,
  Math: OMath, MathRun, MathFraction, MathSubScript, MathSum, MathRadical, MathSuperScript,
} = require("docx");

const FONT = "Times New Roman";
const FIG = path.join(__dirname, "fig");
const DOUBLE = { line: 480, lineRule: "auto" };
const SINGLE = { line: 240, lineRule: "auto" };

// --- inline markup: **bold**, *italic*, ^sup^, ~sub~ ------------------------------------------
function runs(text, base = {}) {
  const out = [];
  for (const part of text.split(/(\*\*[^*]+\*\*|\*[^*]+\*|\^[^^]+\^|~[^~]+~)/)) {
    if (!part) continue;
    if (part.startsWith("**")) out.push(new TextRun({ ...base, text: part.slice(2, -2), bold: true }));
    else if (part.startsWith("*")) out.push(new TextRun({ ...base, text: part.slice(1, -1), italics: true }));
    else if (part.startsWith("^")) out.push(new TextRun({ ...base, text: part.slice(1, -1), superScript: true }));
    else if (part.startsWith("~")) out.push(new TextRun({ ...base, text: part.slice(1, -1), subScript: true }));
    else out.push(new TextRun({ ...base, text: part }));
  }
  return out;
}
const P = (text, o = {}) => new Paragraph({ children: runs(text, o.run || {}), alignment: o.align ?? AlignmentType.JUSTIFIED,
  spacing: { ...DOUBLE, after: 0, ...(o.spacing || {}) }, indent: o.indent ?? { firstLine: 360 }, keepNext: o.keepNext });
const PN = (text, o = {}) => P(text, { ...o, indent: {} }); // no first-line indent (first paragraph of a section)
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t })] });
const bullet = (t) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: runs(t),
  spacing: { ...DOUBLE, after: 0 }, alignment: AlignmentType.LEFT });
const numbered = (t, ref = "contrib") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t),
  spacing: { ...DOUBLE, after: 0 }, alignment: AlignmentType.JUSTIFIED });

// --- tables: Elsevier three-line style, caption above -----------------------------------------
const TW = 9026; // A4 text width in DXA with 1-inch margins
const line = { style: BorderStyle.SINGLE, size: 8, color: "000000" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function table(num, caption, header, rows, widths, notes) {
  const tot = widths.reduce((a, b) => a + b, 0);
  const w = widths.map((x) => Math.round((x / tot) * TW));
  w[w.length - 1] += TW - w.reduce((a, b) => a + b, 0);
  const isNum = (v) => /^[\s*]*[+\-\u2212\u2248<]?\s*[\d.,]+[\s%a-z()\u2013GPU]*\**$|^\u2013$|^pending$/i.test(String(v).replace(/\*\*/g, ""));
  const numCol = header.map((_, i) => i > 0 && rows.every((r) => isNum(r[i])));
  const cell = (t, i, isHead, isLast) => new TableCell({
    width: { size: w[i], type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    borders: { top: isHead ? line : none, bottom: isHead || isLast ? line : none, left: none, right: none },
    children: [new Paragraph({ spacing: SINGLE, alignment: numCol[i] ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: runs(String(t), { size: 20, bold: isHead || undefined }) })],
  });
  const out = [
    new Paragraph({ spacing: { ...SINGLE, before: 240, after: 120 }, keepNext: true, alignment: AlignmentType.LEFT,
      children: [new TextRun({ text: `Table ${num}`, bold: true }), new TextRun({ text: "" }),
        new TextRun({ text: "", break: 1 }), ...runs(caption)] }),
    new Table({ width: { size: TW, type: WidthType.DXA }, columnWidths: w,
      rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true, false)) }),
        ...rows.map((r, ri) => new TableRow({ children: r.map((c, i) => cell(c, i, false, ri === rows.length - 1)) }))] }),
  ];
  if (notes) out.push(new Paragraph({ spacing: { ...SINGLE, before: 60, after: 240 }, children: runs(notes, { size: 18 }) }));
  else out.push(new Paragraph({ spacing: { ...SINGLE, after: 120 }, children: [] }));
  return out;
}

// --- figures: caption below -------------------------------------------------------------------
function figure(num, file, wpx, caption) {
  const buf = fs.readFileSync(path.join(FIG, file));
  const W = buf.readUInt32BE(16), H = buf.readUInt32BE(20);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { ...SINGLE, before: 240 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: wpx, height: Math.round((wpx * H) / W) },
        altText: { title: `Figure ${num}`, description: caption.replace(/\*/g, ""), name: `Figure ${num}` } })] }),
    new Paragraph({ spacing: { ...SINGLE, before: 120, after: 240 }, alignment: AlignmentType.JUSTIFIED,
      children: [new TextRun({ text: `Fig. ${num}. `, bold: true }), ...runs(caption)] }),
  ];
}

// --- display equations, numbered at the right margin -------------------------------------------
const eq = (num, children) => new Paragraph({
  tabStops: [{ type: TabStopType.CENTER, position: TW / 2 }, { type: TabStopType.RIGHT, position: TW }],
  spacing: { ...DOUBLE, before: 60, after: 60 },
  children: [new TextRun("\t"), new OMath({ children }), new TextRun(`\t(${num})`)],
});
const sub = (b, s) => new MathSubScript({ children: [new MathRun(b)], subScript: [new MathRun(s)] });

// ============================================================================================
const TITLE = "Reading the generator’s handwriting: Tokenizer-aware features for fraud detection on LLM-synthesised tabular data";

const HIGHLIGHTS = [
  "LLM-generated tabular data leak labels through GPT-2 number tokens",
  "A funnel test finds token artefacts absent from the NumPy source data",
  "Leak-free encoding of generator units lifts LightGBM to 0.9506 AUC",
  "Three-leaf trees and a GLM match deeper models: the signal is additive",
  "Public-leaderboard gains below 0.0005 AUC are within sampling noise",
];
for (const h of HIGHLIGHTS) if (h.length > 85) throw new Error("Highlight over 85 characters: " + h);

const ABSTRACT = "Synthetic tabular benchmarks produced by language models can carry the class label in the generator’s tokens, and modelling those tokens matters more than tuning the learner. We study Kaggle Playground Series S6E11, credit-card fraud detection on 699,960 synthetic transactions with a 1.67% fraud rate, evaluated by ROC-AUC. A funnel test shows that fraud rates vary across decimal endings and GPT-2 byte-pair tokens far beyond chance: 11% to 50% of value groups fall outside ±2σ of the overall rate, against 0% to 5% with shuffled labels or in the NumPy-simulated source data. We therefore re-express every number in the units GPT-2 emitted (positional digits, integer and decimal parts, and tokens), encode those units with leak-free in-fold target encoding and neighbour-value frequencies, and fit deliberately shallow models. On one fixed five-fold split, a single three-leaf LightGBM improves from 0.94674 to 0.95063 out-of-fold AUC. A model restricted to decimal digits and token tails reaches 0.611 AUC on its own. A logistic generalised linear model on the same features scores 0.95031, indicating that the signal is nearly additive. A logit-rank hill-climbing ensemble of shallow gradient-boosted trees, the linear model and a token-embedding neural network reaches 0.95099 under nested cross-validation, and public-leaderboard scores tracked validation to within 0.0006. Subsampling experiments show that public-leaderboard differences below about 0.0005 AUC are within noise, which argues against leaderboard probing. Generator-aware forensics of this kind should be a routine check on synthetic training and benchmark data.";
const nWords = ABSTRACT.split(/\s+/).length;
if (nWords > 250) throw new Error("Abstract has " + nWords + " words");

const KEYWORDS = "Synthetic tabular data; Fraud detection; Large language models; Target encoding; Gradient boosting; Data leakage";

const body = [];
const add = (...xs) => xs.flat().forEach((x) => body.push(x));

// ---------- 1 Introduction ----------
add(H1("1. Introduction"),
  PN("Synthetic tabular data are increasingly used to share sensitive records, to augment scarce classes and to build public benchmarks. Kaggle’s Playground Series releases a monthly competition whose data are sampled from a deep generative model trained on a small real or simulated dataset. Competitors usually treat these rows as ordinary data and tune gradient-boosted decision trees (GBDTs), which remain the strongest general-purpose learners on tabular problems (Grinsztajn et al., 2022). In Season 6 Episode 11 (S6E11), fraud detection on 700,000 transactions, that approach plateaus near 0.947 ROC-AUC, whereas careful competitors reach 0.951 with the same algorithms."),
  P("We show that the gap is explained by where the generator stores the label. The rows were written by a fine-tuned GPT-2 (Radford et al., 2019) in the manner of GReaT (Borisov et al., 2023), and the fraud label correlates with the textual units that GPT-2 emitted: decimal endings and byte-pair tokens. A tree that splits numbers on magnitude cannot see these units, because adjacent values such as 12.69 and 12.70 belong to different units. Exposing the units as features, encoded without label leakage, closes most of the gap. The phenomenon is a form of shortcut learning (Geirhos et al., 2020) and of data leakage in the sense of Kaufman et al. (2012): the label is predictable from an artefact of how the data were produced rather than from the domain mechanism."),
  P("The paper makes five contributions:"),
  numbered("A model-free forensic test showing that the label signal in decimal endings and GPT-2 tokens exists in the language-model-generated data and is absent from the NumPy-simulated source data (Section 4)."),
  numbered("A feature recipe that mirrors how the generator writes numbers: positional digits, leak-free target encodings of exact values, integer parts, decimal parts and byte-pair tokens, and neighbour-value frequency features (Section 5)."),
  numbered("An ablation ladder on one fixed five-fold split that attributes the 0.0039 AUC single-model gain to its components (Section 6)."),
  numbered("Evidence that the signal is nearly additive: three-leaf trees outperform deep trees, and a logistic generalised linear model (GLM) on the same features matches GBDTs to within 0.0003 AUC (Section 6)."),
  numbered("A diverse ensemble of shallow GBDTs, a GLM and a token-embedding neural network, selected and scored by nested cross-validation rather than by public-leaderboard feedback, together with a quantification of leaderboard noise (Sections 6 and 8)."),
);

// ---------- 2 Related work ----------
add(H1("2. Related work"),
  PN("**Synthetic tabular data.** Deep generators for tables include conditional GANs such as CTGAN (Xu et al., 2019) and language-model approaches such as GReaT, which serialise each row as a sentence of the form “column is value”, fine-tune a pretrained language model on these sentences and sample new rows token by token (Borisov et al., 2023). Evaluation of such generators usually focuses on marginal fidelity, pairwise correlations and downstream utility. Systematic, label-dependent artefacts introduced by tokenisation have received far less attention."),
  P("**Gradient boosting on tabular data.** LightGBM (Ke et al., 2017), XGBoost (Chen & Guestrin, 2016) and CatBoost (Prokhorenkova et al., 2018) dominate tabular benchmarks (Grinsztajn et al., 2022). High-cardinality categorical keys are commonly handled by smoothed target encoding (Micci-Barreca, 2001), which must be fitted out-of-fold to avoid target leakage. Neural alternatives embed categorical fields and combine a linear “wide” part with a deep part (Cheng et al., 2016)."),
  P("**Ensembling and leaderboards.** Greedy forward selection with replacement (Caruana et al., 2004) is a standard way to weight heterogeneous models. Kaggle competitions score a public subset of the test data during the competition and a disjoint private subset at the end. Repeatedly selecting submissions on the public subset overfits it; public notebooks in this competition illustrate the practice (goodpjw2008, 2026). Closest to our work, a public notebook in the same competition observed that GPT-2 tokens of two columns carry signal and reported a 10-fold pipeline (azizdzhon, 2026). We extend that observation with a model-free test against shuffled labels and the source data, an ablation ladder, an artefact-only model, a linear-model comparison and a leaderboard-noise analysis."),
);

// ---------- 3 Data ----------
add(H1("3. Data and task"),
  PN("The task is binary classification of 299,984 test transactions as fraudulent or legitimate, scored by ROC-AUC, using 699,960 labelled training rows with a 1.67% fraud rate (Kaggle, 2026). The public leaderboard is computed on about 20% of the test set; the private leaderboard, which determines the final ranking, uses the remaining 80%. Table 1 lists the 24 features."),
  table(1, "Feature groups of the S6E11 dataset.", ["Feature group", "Count", "Columns"], [
    ["Continuous, two decimals", "4", "amount_usd, account_balance_usd, hours_since_last_txn, distance_from_home_km"],
    ["Continuous, one decimal", "2", "velocity_score, merchant_risk_score"],
    ["Integer counts and calendar", "7", "txn_count_last_24h, card_age_months, customer_age, cvv_retry_count, time_of_day_hour, day_of_week, prior_disputes"],
    ["Categorical", "5", "merchant_category (12 levels), card_type (5), auth_method (5), channel (5), device_type (8)"],
    ["Boolean flags", "6", "is_foreign_transaction, is_new_merchant, used_vpn, ip_country_mismatch, billing_shipping_mismatch, is_ai_generated_scam_attempt"],
  ], [2.2, 0.8, 6]),
  P("The competition data were generated from the public Credit Card Fraud Detection 2026 dataset (uditjain13, 2026): 20,000 rows with 339 frauds (1.70%), released under CC0 and itself simulated with NumPy. No competition row matches an original row exactly, yet 85% to 100% of each numeric column’s values also occur in the original. The strongest single raw signals are merchant_risk_score (univariate AUC 0.749), velocity_score (0.711) and cvv_retry_count (0.708). Authentication matters sharply: transactions without authentication are fraudulent at 6.4%, against 0.5% for biometric authentication. The fraud rate shows no drift across row identifiers, so shuffled cross-validation is appropriate."),
  P("**Validation protocol.** Every model uses the same stratified five-fold split (shuffled, seed 42). All supervised encodings are fitted inside each training fold, with an inner five-fold split for the training rows themselves, so out-of-fold (OOF) AUC is an unbiased estimate. Because all models share the folds, differences between them are paired. Ensembles are scored by nested cross-validation: weights are fitted on four folds and evaluated on the fifth."),
);

// ---------- 4 Forensics ----------
add(H1("4. Generator forensics"),
  PN("A GReaT-style generator writes a row as text, and GPT-2’s byte-level byte-pair encoder cuts a number at the decimal point and then splits the integer according to its merge table, not by arithmetic (Table 2). Every account balance in the training set has an integer part of two or more tokens, against 7.1% of amounts. The decimal ending and the second integer token are therefore units that the generator chose separately, and each choice could carry information about the label sampled in the same row."),
  table(2, "How GPT-2 tokenises numbers written by the generator.", ["Value", "GPT-2 tokens"], [
    ["61.16", "“ 61” · “.” · “16”"],
    ["1269.35", "“ 12” · “69” · “.” · “35”"],
    ["3233.0", "“ 3” · “233” · “.” · “0”"],
  ], [2, 5]),
  P("**Funnel test.** For each unit we group rows by its value g, keep groups with at least 100 rows (30 in the smaller original data), and count the groups whose fraud rate p̂~g~ departs from the overall rate p̄ by more than two binomial standard deviations:"),
  eq(1, [new MathRun("|"), sub("p̂", "g"), new MathRun(" − p̄| > 2"),
    new MathRadical({ children: [new MathFraction({ numerator: [new MathRun("p̄(1 − p̄)")], denominator: [sub("n", "g")] })] })]),
  P("With no signal about 4.6% of groups exceed this band by chance. We compare real labels, labels shuffled five times, and the original NumPy-generated data, and report a χ² test of independence between unit value and label."),
  figure(1, "fig1_forensics.png", 590, "Funnel test per generator unit. Each row is a unit (the decimal part of a column, or the second GPT-2 token of its integer part). Points show the share of value groups whose fraud rate lies outside ±2σ of the overall rate. Blue: competition data with real labels; open circles: competition data with shuffled labels (mean of five permutations); grey squares: original NumPy-simulated data. The dashed line marks the chance level. The original data have no amount groups with two tokens."),
  P("In the competition data 11% to 50% of groups leave the band, with χ² p-values as small as 10^−100^ for the decimal part of hours since last transaction (Fig. 1). Shuffled labels return every unit to chance. In the original data, which were simulated without a language model, no unit shows signal: every χ² p-value exceeds 0.09. The artefact is therefore introduced by the generator, not by the fraud mechanism."),
);

// ---------- 5 Method ----------
add(H1("5. Method"),
  PN("The method rewrites each row in the generator’s own units, encodes those units without label leakage and fits deliberately simple models on them (Fig. 2)."),
  figure(2, "fig2_pipeline.png", 560, "Modelling pipeline. Every feature mirrors a unit that the generator wrote, and every supervised encoder is fitted inside the training fold. Three model families with different inductive biases are blended in logit-rank space."),
  H2("5.1. Generator-unit decomposition"),
  PN("For the four two-decimal columns and the two one-decimal columns we extract the last two integer digits, each decimal digit, the integer part and the decimal part. Customer age and card age contribute their last digit. For account balance and amount we look up the first and second GPT-2 token of the integer part, and the second token combined with the decimals. These features use no labels."),
  H2("5.2. Leak-free target encoding"),
  PN("Each key k (an exact value, an integer part, a decimal part or a token) is replaced by its smoothed fraud rate (Micci-Barreca, 2001):"),
  eq(2, [new MathRun("TE(k) = "), new MathFraction({
    numerator: [new MathSum({ children: [sub("y", "i")], subScript: [new MathRun("i: "), sub("k", "i"), new MathRun(" = k")] }), new MathRun(" + mȳ")],
    denominator: [sub("n", "k"), new MathRun(" + m")] }), new MathRun(",   m = 20")]),
  P("where n~k~ is the number of training rows with key k and ȳ is the training-fold fraud rate. Rows of a validation fold or of the test set are encoded with statistics from the whole training fold. Rows of the training fold are encoded with an inner five-fold split, so that no row ever sees its own label."),
  H2("5.3. Neighbour-value frequencies"),
  PN("For each continuous column we measure how crowded a value is: its count in the training fold, the mean count of its one and five nearest distinct values on each side, the log-ratio between the two (a spike indicator) and the gaps to the adjacent observed values. These label-free features tell a model how much to trust an exact-value encoding."),
  H2("5.4. Models"),
  PN("Because the signal is nearly additive (Section 6), trees are kept extremely shallow. LightGBM uses three leaves at depth three with at least 1,700 rows per leaf, a learning rate of 0.02 and three seeds; XGBoost uses depth two and CatBoost depth three. A logistic GLM on the logit of the encodings, cubic splines of the numeric columns, one-hot categories and the frequency block provides a linear view. A wide-and-deep network (Cheng et al., 2016) embeds 33 generator-unit fields without any target encoding and adds quantile-normalised numerics; it is regularised with embedding dropout, an L2 penalty on the embeddings in each batch and a minimum of 50 rows per level, and is trained for six epochs with five seeds on an Apple M5 GPU (Paszke et al., 2019)."),
  H2("5.5. Blending"),
  PN("Predictions are converted to logit-ranks, which removes calibration differences between model families:"),
  eq(3, [new MathRun("L(v) = ln "), new MathFraction({ numerator: [new MathRun("r(v) − 0.5")], denominator: [new MathRun("n − r(v) + 0.5")] })]),
  P("where r(v) is the rank of prediction v among n predictions. Weights are obtained by greedy forward selection with replacement (Caruana et al., 2004). The ensemble is scored by nested cross-validation and is never chosen by public-leaderboard feedback."),
);

// ---------- 6 Results ----------
add(H1("6. Experiments and results"),
  PN("Generator-aware features raise a single LightGBM from 0.94674 to 0.95063 OOF AUC, and the ensemble reaches 0.95099 (Fig. 3). The two largest steps are making the trees shallow (+0.00127) and target-encoding exact values, integer parts and decimal parts inside each fold (+0.00155). Digits, frequencies and tokens add smaller but consistent amounts. From shallow trees onward every feature step improved all five folds; three seeds at the lower learning rate improved four of five. Fold-level scores were not retained for the deep-tree starting point."),
  figure(3, "fig3_ablation.png", 590, "Ablation ladder. Each green bar is the out-of-fold AUC gain of one step over the previous step, in units of 10^−4^; the grey bar is the total. All steps use the same five folds. The final step is the nested cross-validated ensemble score."),
  P("**Artefact-only model.** A LightGBM restricted to decimal digits and the encodings of decimal parts and token tails, with no magnitudes, scores 0.6112 AUC (fold range 0.600–0.620). This is the part of the label that the generator wrote into its units, separate from the fraud logic encoded in the magnitudes."),
  P("**Model zoo.** Table 3 compares the ensemble members. The GLM is nearly as accurate as the best GBDT, which supports the additivity interpretation. The blend assigns weights of 0.43 to LightGBM (two runs), 0.29 to the GLM, 0.14 to CatBoost and 0.14 to the token network; XGBoost is not selected. The GLM and the network earn their weight through different errors rather than higher accuracy."),
  table(3, "Ensemble members on generator-aware features (five folds, same split).", ["Model", "Inductive bias", "OOF AUC", "Training time"], [
    ["LightGBM, tokens + frequencies, 3 seeds", "Shallow trees", "0.95063", "12 min"],
    ["Logistic GLM", "Additive, linear in logit TE", "0.95031", "25 s"],
    ["CatBoost, depth 3", "Symmetric shallow trees", "0.94970", "37 min"],
    ["Token-embedding network, 5 seeds", "Learned embeddings, no TE", "0.94945", "≈3 min (GPU)"],
    ["XGBoost, depth 2, tokens + frequencies", "Shallow trees", "0.94898", "3 min"],
    ["**Logit-rank hill-climbing ensemble**", "Ensemble", "**0.95099** (nested)", "seconds"],
  ], [3.2, 2.6, 1.6, 1.4], "*Note.* Times are wall-clock on one Apple M5 laptop (10 CPU cores, 16 GB) for five folds, measured while other jobs shared the machine. TE = target encoding."),
  P("**Leaderboard.** Table 4 reports the submissions made during the study. Public scores tracked cross-validated scores closely, and in the same order."),
  table(4, "Cross-validated and public-leaderboard AUC of the submissions.", ["Submission", "Content", "CV AUC", "Public AUC"], [
    ["v1", "Deep LightGBM + XGBoost, raw features", "0.94724", "0.94754"],
    ["v2", "Shallow GBDTs with digits, TE, tokens, frequencies", "0.95069", "0.95126"],
    ["v3", "v2 + GLM + token-embedding network", "0.95099", "pending"],
  ], [1.2, 4.6, 1.4, 1.4]),
);

// ---------- 7 Negative results ----------
add(H1("7. Negative results"),
  PN("Neither the original source dataset nor conventional domain features improved the model (Table 5). The original data cannot help because their rows carry no generator artefacts (Section 4), and the relationship they encode is already learned from 700,000 synthetic rows. The domain features re-express magnitudes on which the trees already split and add noise. A public study of this competition reports the same pattern for pairs of exact values and for profile-level encodings (azizdzhon, 2026)."),
  table(5, "Ideas that did not improve out-of-fold AUC.", ["Idea", "OOF AUC", "Change"], [
    ["Original 20,000 rows appended to each training fold", "0.94673", "−0.00001"],
    ["Smoothed fraud rate per value, measured on the original data", "0.94648", "−0.00026"],
    ["Score of a logistic model fitted on the original data", "0.94674", "0.00000"],
    ["All three together", "0.94679", "+0.00005"],
    ["30 hand-built domain features, deep LightGBMᵃ", "0.94568", "−0.00132"],
    ["The same domain features, XGBoostᵃ", "0.94523", "−0.00150"],
    ["Spline logistic regression on raw featuresᵃ", "0.94340", "−0.00360"],
  ], [5.6, 1.4, 1.4], "*Note.* The first four rows are measured against a fast LightGBM (learning rate 0.05) at 0.94674. ᵃ Measured against the matching raw-feature model (LightGBM 0.94700 or XGBoost 0.94673, learning rate 0.02). Domain features: ratios, transactions per hour, risk × velocity, group-relative values, frequency counts and category crosses."),
);

// ---------- 8 Discussion ----------
add(H1("8. Discussion"),
  H2("8.1. The public leaderboard cannot rank near-identical models"),
  PN("We drew 400 random 60,000-row subsets of the training set, the size of the public test portion, and scored the out-of-fold predictions on each (Table 6). A single public score varies by about ±0.003. The ensemble’s 0.00040 advantage over the best single model appears with the wrong sign on 6% of subsets, whereas the 0.0037 gain of the generator-aware model over the raw baseline never does."),
  table(6, "AUC on 400 random 60,000-row subsets of the out-of-fold predictions.", ["Quantity", "Mean", "SD", "Wrong sign"], [
    ["AUC of the final ensemble", "0.95116", "0.00292", "–"],
    ["Ensemble minus best single model", "+0.00040", "0.00026", "6%"],
    ["Best single model minus raw-feature baseline", "+0.00367", "0.00090", "< 0.3%"],
  ], [4.4, 1.4, 1.3, 1.4]),
  P("At the time of writing, the public top of this competition is a chain of notebooks that re-blend one another’s submissions in steps of 0.00001, guided by a regression on more than 200 scored public files (goodpjw2008, 2026). Those steps are about 20 times smaller than the paired noise in Table 6, so they mostly fit the 20% public slice. We therefore selected every component and weight by nested cross-validation."),
  H2("8.2. Irreducible noise and additivity"),
  PN("Even on the original simulated data, a well-specified logistic model reaches only 0.944 cross-validated AUC, because fraud is drawn at random given the features; very high targets such as 0.98 are not attainable. Three-leaf trees beat 31-leaf trees by 0.0013 on raw features, and a GLM matches GBDTs to within 0.0003 on the full feature set. The label effect of each generator unit appears to add up across columns, which is consistent with a generator that writes one column after another."),
  H2("8.3. Implications for practice"),
  PN("Practitioners who train or benchmark on synthetic tabular data should identify the generator and the units it writes before modelling; for language-model generators these are tokens, not values. The funnel test of Section 4 needs no model and runs in seconds, and its comparison against shuffled labels and the source data separates generator artefacts from domain signal. Where artefacts are present, models evaluated on such data may reward shortcut features that do not transfer to real transactions, so benchmark designers should report artefact-only baselines alongside headline scores. Competitors should encode high-cardinality artefact keys strictly inside folds, prefer additive or shallow learners, and measure leaderboard noise before acting on it."),
  H2("8.4. Limitations"),
  PN("All results come from one competition and one generator family. The token analysis covers only the integer parts of two columns. We did not train a language-model classifier on the serialised rows, which a public study reports as a weak but complementary ensemble member (azizdzhon, 2026). Private-leaderboard results were not available at the time of writing."),
);

// ---------- 9 Conclusion ----------
add(H1("9. Conclusion"),
  PN("When a language model writes the data, its tokens carry the signal. Re-reading each number as the units GPT-2 emitted, encoding those units without leakage and fitting shallow additive models lifted a single model from 0.94674 to 0.95063 out-of-fold AUC on Kaggle Playground S6E11, and an ensemble of shallow GBDTs, a GLM and a token-embedding network reached 0.95099 under nested cross-validation. The same forensic test found no such signal in the NumPy-simulated source data, so the gain comes from the generator rather than from fraud itself. As more training sets and benchmarks are produced by language models, generator-aware validation of this kind should become a routine check, because it detects shortcuts that models will otherwise learn without notice."),
);

// ---------- Declarations ----------
const decl = (title, text) => [new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(title)] }), PN(text)];
add(
  decl("CRediT authorship contribution statement", "**[Author name]:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing – original draft, Writing – review & editing."),
  decl("Declaration of competing interest", "The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper."),
  decl("Data availability", "The competition data are available from Kaggle under the competition rules (Kaggle, 2026). The original dataset is available under CC0 (uditjain13, 2026). The analysis code is available from the corresponding author on request [replace with repository link]."),
  decl("Declaration of generative AI and AI-assisted technologies in the manuscript preparation process", "During the preparation of this work the author(s) used Claude (Anthropic) to write analysis code, run experiments and draft the manuscript. After using this tool, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the published article."),
  decl("Funding", "This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors."),
);

// ---------- References (APA 7, alphabetical) ----------
const REFS = [
  "azizdzhon. (2026). *How GReaT tokenizes numbers: BPE-token target encoding & 10-fold pipeline* [Kaggle notebook]. https://www.kaggle.com/code/azizdzhon/how-great-tokenizes-numbers-bpe-token-target",
  "Borisov, V., Seßler, K., Leemann, T., Pawelczyk, M., & Kasneci, G. (2023). Language models are realistic tabular data generators. In *Proceedings of the Eleventh International Conference on Learning Representations (ICLR 2023)*. https://arxiv.org/abs/2210.06280",
  "Caruana, R., Niculescu-Mizil, A., Crew, G., & Ksikes, A. (2004). Ensemble selection from libraries of models. In *Proceedings of the 21st International Conference on Machine Learning* (p. 18). ACM. https://doi.org/10.1145/1015330.1015432",
  "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). https://doi.org/10.1145/2939672.2939785",
  "Cheng, H.-T., Koc, L., Harmsen, J., Shaked, T., Chandra, T., Aradhye, H., Anderson, G., Corrado, G., Chai, W., Ispir, M., Anil, R., Haque, Z., Hong, L., Jain, V., Liu, X., & Shah, H. (2016). Wide & deep learning for recommender systems. In *Proceedings of the 1st Workshop on Deep Learning for Recommender Systems* (pp. 7–10). https://doi.org/10.1145/2988450.2988454",
  "Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence, 2*(11), 665–673. https://doi.org/10.1038/s42256-020-00257-z",
  "goodpjw2008. (2026). *S6E11 | Board model + RealMLP* [Kaggle notebook]. https://www.kaggle.com/code/goodpjw2008/s6e11-board-model-realmlp-lb-0-95562",
  "Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? In *Advances in Neural Information Processing Systems 35, Datasets and Benchmarks Track* (pp. 507–520).",
  "Kaggle. (2026). *Playground Series – Season 6, Episode 11: Predicting credit card fraud* [Data set and competition]. https://www.kaggle.com/competitions/playground-series-s6e11",
  "Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data, 6*(4), Article 15. https://doi.org/10.1145/2382577.2382579",
  "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems 30* (pp. 3146–3154).",
  "Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. *ACM SIGKDD Explorations Newsletter, 3*(1), 27–32. https://doi.org/10.1145/507533.507538",
  "Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., … Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In *Advances in Neural Information Processing Systems 32* (pp. 8024–8035).",
  "Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. In *Advances in Neural Information Processing Systems 31* (pp. 6638–6648).",
  "Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language models are unsupervised multitask learners* [Technical report]. OpenAI.",
  "uditjain13. (2026). *Credit card fraud detection 2026* [Data set]. Kaggle. https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026",
  "Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). Modeling tabular data using conditional GAN. In *Advances in Neural Information Processing Systems 32* (pp. 7335–7345).",
];
add(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("References")] }),
  REFS.map((r) => new Paragraph({ children: runs(r), alignment: AlignmentType.LEFT, spacing: { ...DOUBLE, after: 0 },
    indent: { left: 720, hanging: 720 } })));

// ---------- Appendix ----------
add(new Paragraph({ children: [new PageBreak()] }),
  H1("Appendix A. Reproducibility"),
  PN("All code runs on one Apple M5 laptop (10 CPU cores, 16 GB memory) with Python 3.14, LightGBM 4.7, XGBoost 3.4, CatBoost, scikit-learn and PyTorch 2.15 with the Metal (MPS) backend. The GPT-2 tokenizer files were obtained from the Hugging Face openai-community/gpt2 repository and checked by asserting that “ 1269.35” splits into “ 12”, “69”, “.” and “35”. Table A.1 lists the scripts."),
  table("A.1", "Analysis scripts.", ["Script", "Role"], [
    ["src/common.py", "Data loading and the fixed five-fold split (stratified, seed 42)"],
    ["src/feats3.py", "Generator-unit features, in-fold target encoding, neighbour frequencies"],
    ["src/forensics.py", "Funnel test (Section 4)"],
    ["src/exp3.py, src/exp_orig.py", "Ablations and original-data experiments"],
    ["src/train3.py", "Shallow LightGBM, XGBoost and CatBoost"],
    ["src/glm.py", "Additive logistic model"],
    ["src/nn_tok.py", "Token-embedding network"],
    ["src/ensemble.py", "Logit-rank hill climbing with nested cross-validation"],
  ], [2.6, 6]),
);

// ============================================================================================
const titlePage = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { ...DOUBLE, after: 240 },
    children: [new TextRun({ text: TITLE, bold: true, size: 32 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: DOUBLE,
    children: [new TextRun("[Author name]"), new TextRun({ text: "a,", superScript: true }), new TextRun({ text: "*", superScript: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: DOUBLE,
    children: [new TextRun({ text: "a", superScript: true }), new TextRun({ text: " [Department, Institution, City, Postal code, Country]", italics: true })] }),
  new Paragraph({ spacing: { ...DOUBLE, before: 480 }, children: [new TextRun({ text: "*", superScript: true }),
    new TextRun(" Corresponding author. E-mail address: [email address]. ORCID: [0000-0000-0000-0000].")] }),
  new Paragraph({ spacing: DOUBLE, children: runs("*Article type:* Research article. *Word count of abstract:* " + nWords + ".") }),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Highlights")] }),
  ...HIGHLIGHTS.map(bullet),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Abstract")] }),
  PN(ABSTRACT),
  new Paragraph({ spacing: { ...DOUBLE, before: 240 }, children: [new TextRun({ text: "Keywords: ", italics: true }), new TextRun(KEYWORDS)] }),
  new Paragraph({ children: [new PageBreak()] }),
];

const doc = new Document({
  creator: "[Author name]", title: TITLE, description: "Manuscript prepared in the Expert Systems with Applications format",
  styles: {
    default: { document: { run: { font: FONT, size: 24 }, paragraph: { spacing: DOUBLE } } },
    characterStyles: [{ id: "LineNumber", name: "line number", basedOn: "DefaultParagraphFont", run: { font: FONT, size: 20 } }],
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true }, paragraph: { spacing: { ...DOUBLE, before: 240, after: 0 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, italics: true }, paragraph: { spacing: { ...DOUBLE, before: 120, after: 0 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { run: { font: FONT }, paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    { reference: "contrib", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "(%1)", alignment: AlignmentType.LEFT,
      style: { run: { font: FONT }, paragraph: { indent: { left: 720, hanging: 432 } } } }] },
  ] },
  sections: [{
    properties: {
      page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } },
      lineNumbers: { countBy: 1, restart: LineNumberRestartFormat.CONTINUOUS, distance: 360 },
    },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, spacing: SINGLE,
      children: [new TextRun({ children: [PageNumber.CURRENT] })] })] }) },
    children: [...titlePage, ...body],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "Generator_Handwriting_ESWA_manuscript.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, "| abstract words:", nWords, "| highlights:", HIGHLIGHTS.map((h) => h.length).join(","));
});
