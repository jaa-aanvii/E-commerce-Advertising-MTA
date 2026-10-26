// Generates the IEEE-style research paper as a .docx (two-column, Letter, Times New Roman).
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, BorderStyle,
  AlignmentType, SectionType, TabStopType, Tab, HighlightColor, Footer, PageNumber, VerticalAlign,
  Math: OMath, MathRun, MathFraction, MathSum, MathSubScript, MathSuperScript, MathSubSuperScript,
  MathRoundBrackets, MathSquareBrackets, LevelFormat, ShadingType,
} = require("docx");

const OUT = process.argv[2] || "paper.docx";
const FONT = "Times New Roman";
const COL = 5040; // column width in DXA (two columns + 360 gap on a 10440 text block)

// ------------------------------------------------------------------ inline markup
// **bold**  *italic*  {{placeholder}}  _{sub}  ^{sup}
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\{\{[^}]+\}\}|_\{[^}]+\}|\^\{[^}]+\})/g;
  let last = 0, m;
  const push = (t, o) => { if (t) out.push(new TextRun({ text: t, font: FONT, ...base, ...o })); };
  while ((m = re.exec(text)) !== null) {
    push(text.slice(last, m.index), {});
    const tok = m[0];
    if (tok.startsWith("**")) push(tok.slice(2, -2), { bold: true });
    else if (tok.startsWith("{{")) push(tok.slice(2, -2), { highlight: HighlightColor.YELLOW, bold: true });
    else if (tok.startsWith("_{")) push(tok.slice(2, -1), { subScript: true });
    else if (tok.startsWith("^{")) push(tok.slice(2, -1), { superScript: true });
    else push(tok.slice(1, -1), { italics: true });
    last = m.index + tok.length;
  }
  push(text.slice(last), {});
  return out;
}

const body = [];
const P = (text, o = {}) => body.push(new Paragraph({
  children: runs(text, { size: 20 }), alignment: AlignmentType.JUSTIFIED,
  indent: o.noIndent ? undefined : { firstLine: 202 }, spacing: { after: 0, line: 240 }, ...o.p,
}));
const H1 = (num, text) => body.push(new Paragraph({
  children: [new TextRun({ text: `${num}. ${text}`, font: FONT, size: 20, smallCaps: true })],
  alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 }, keepNext: true,
}));
const H2 = (letter, text) => body.push(new Paragraph({
  children: [new TextRun({ text: `${letter}. ${text}`, font: FONT, size: 20, italics: true })],
  spacing: { before: 120, after: 60 }, keepNext: true,
}));
const BUL = (text) => body.push(new Paragraph({
  children: runs(text, { size: 20 }), alignment: AlignmentType.JUSTIFIED,
  numbering: { reference: "bullets", level: 0 }, spacing: { after: 20 },
}));

const thin = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };

// ------------------------------------------------------------------ math helpers
const R = (s) => new MathRun(s);
const sub = (b, s) => new MathSubScript({ children: typeof b === "string" ? [R(b)] : b, subScript: typeof s === "string" ? [R(s)] : s });
const sup = (b, s) => new MathSuperScript({ children: typeof b === "string" ? [R(b)] : b, superScript: typeof s === "string" ? [R(s)] : s });
const subsup = (b, s, p) => new MathSubSuperScript({ children: typeof b === "string" ? [R(b)] : b, subScript: [R(s)], superScript: [R(p)] });
const frac = (n, d) => new MathFraction({ numerator: n, denominator: d });
const sum = (children, s, p) => new MathSum({ children, subScript: s ? (typeof s === "string" ? [R(s)] : s) : undefined, superScript: p ? [R(p)] : undefined });
const rb = (children) => new MathRoundBrackets({ children });
const sb = (children) => new MathSquareBrackets({ children });
let eqNo = 0;
const EQ = (children) => {
  eqNo += 1;
  const cell = (w, kids, align) => new TableCell({
    width: { size: w, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    margins: { top: 40, bottom: 40, left: 0, right: 0 },
    borders: { top: none, bottom: none, left: none, right: none },
    children: [new Paragraph({ children: kids, alignment: align })],
  });
  body.push(new Table({
    width: { size: COL, type: WidthType.DXA }, columnWidths: [500, COL - 1000, 500],
    rows: [new TableRow({ cantSplit: true, children: [
      cell(500, [], AlignmentType.LEFT),
      cell(COL - 1000, [new OMath({ children })], AlignmentType.CENTER),
      cell(500, [new TextRun({ text: `(${eqNo})`, font: FONT, size: 20 })], AlignmentType.RIGHT),
    ] })],
  }));
  return eqNo;
};

// ------------------------------------------------------------------ tables
const ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII"];
let tabNo = 0;
function TABLE(title, header, rows, widths, opts = {}) {
  tabNo += 1;
  body.push(new Paragraph({
    children: [new TextRun({ text: `TABLE ${ROMAN[tabNo - 1]}`, font: FONT, size: 16, smallCaps: true })],
    alignment: AlignmentType.CENTER, spacing: { before: 140, after: 0 }, keepNext: true,
  }));
  body.push(new Paragraph({
    children: [new TextRun({ text: title, font: FONT, size: 16, smallCaps: true })],
    alignment: AlignmentType.CENTER, spacing: { after: 60 }, keepNext: true,
  }));
  const total = widths.reduce((a, b) => a + b, 0);
  const mk = (cells, isHead, isLast) => new TableRow({
    tableHeader: isHead, cantSplit: true,
    children: cells.map((c, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      margins: { top: 20, bottom: 20, left: 60, right: 60 },
      verticalAlign: VerticalAlign.CENTER,
      borders: { top: isHead ? thin : none, bottom: (isHead || isLast) ? thin : none, left: none, right: none },
      children: [new Paragraph({
        children: runs(String(c), { size: 15, bold: isHead ? true : undefined }),
        alignment: (i === 0 || opts.leftAll) ? AlignmentType.LEFT : AlignmentType.CENTER,
        keepNext: !isLast,
      })],
    })),
  });
  body.push(new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER,
    rows: [mk(header, true, false), ...rows.map((r, i) => mk(r, false, i === rows.length - 1))],
  }));
  if (opts.note) body.push(new Paragraph({ children: runs(opts.note, { size: 14 }), spacing: { before: 30, after: 100 }, alignment: AlignmentType.JUSTIFIED }));
  else body.push(new Paragraph({ children: [], spacing: { after: 60 } }));
}

// ------------------------------------------------------------------ figures
let figNo = 0;
function pngSize(file) { const b = fs.readFileSync(file); return [b.readUInt32BE(16), b.readUInt32BE(20)]; }
function FIG(file, caption, widthPx = 318) {
  figNo += 1;
  const [w, h] = pngSize(file);
  body.push(new Paragraph({
    children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: widthPx, height: Math.round(widthPx * h / w) } })],
    alignment: AlignmentType.CENTER, spacing: { before: 100, after: 40 }, keepNext: true,
  }));
  body.push(new Paragraph({ children: runs(`Fig. ${figNo}.  ${caption}`, { size: 16 }), alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 } }));
}
function FIG_PLACEHOLDER(caption, what) {
  figNo += 1;
  body.push(new Paragraph({
    children: runs(`{{[FIGURE TO BE INSERTED: ${what}]}}`, { size: 16 }), alignment: AlignmentType.CENTER,
    border: { top: thin, bottom: thin, left: thin, right: thin }, spacing: { before: 100, after: 40 }, keepNext: true,
  }));
  body.push(new Paragraph({ children: runs(`Fig. ${figNo}.  ${caption}`, { size: 16 }), alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 } }));
}

function ALGO(title, lines) {
  const cellParas = [new Paragraph({ children: [new TextRun({ text: title, font: FONT, size: 16, bold: true })], spacing: { after: 40 } })];
  lines.forEach((l, i) => cellParas.push(new Paragraph({
    children: [new TextRun({ text: `${String(i + 1).padStart(2, " ")}: ${l}`, font: "Courier New", size: 14 })],
    spacing: { after: 0 },
  })));
  body.push(new Table({
    width: { size: COL, type: WidthType.DXA }, columnWidths: [COL],
    rows: [new TableRow({ children: [new TableCell({
      width: { size: COL, type: WidthType.DXA }, margins: { top: 60, bottom: 60, left: 80, right: 80 },
      borders: { top: thin, bottom: thin, left: none, right: none }, children: cellParas,
    })] })],
  }));
  body.push(new Paragraph({ children: [], spacing: { after: 100 } }));
}

const PH = (s) => `{{[TO BE FILLED: ${s}]}}`;
const CH = ["Direct Traffic", "Display Ads", "Email", "Referral", "Search Ads", "Social Media"];
const BUILD = __dirname;

// ================================================================== ABSTRACT
body.push(new Paragraph({
  children: [
    new TextRun({ text: "Abstract", font: FONT, size: 18, bold: true, italics: true }),
    new TextRun({ text: "—Customers of e-commerce businesses usually interact with several advertising channels before they purchase, and assigning credit for a conversion across those touchpoints, known as multi-touch attribution, remains a methodological challenge. Rule-based heuristics such as first-touch, last-touch and linear attribution assign credit by position rather than by modelled contribution. This paper presents an auditable, end-to-end pipeline that reconstructs customer journeys from an advertising interaction log, attributes conversions with two data-driven methods, namely the removal effect of a first-order absorbing Markov chain and exact Shapley values of a coalition game whose value function is the Markov conversion probability, combines their normalized scores into a configurable hybrid score, and converts that score into a whole-rupee, contribution-based budget split. Every stage has explicit validation and edge-case policies and is covered by an automated test suite (473 tests, all passing). Applied to a public log of 10,000 events from 2,847 users recorded over 48 hours across six channels, the pipeline reconstructs 6,382 journeys (4,944 converting and 1,438 non-converting; mean 1.57 touches per journey, 61.5% single-touch) without rejecting any row, and a 6-hour attribution window excludes 24.6% of touches. We also show that, under the implemented value function, the Markov removal effect is a normalized leave-one-out marginal contribution of the same game whose average marginal contribution is the Shapley value. Because the data contain no spend, cost or revenue, the allocation is a transparent contribution-based baseline rather than an ROI optimizer, and the attribution scores are model-based rather than causal. ", font: FONT, size: 18, bold: true }),
    ...runs("{{[Optional: add one sentence summarizing the verified attribution and ranking findings once Section VII is complete.]}}", { size: 18, bold: true }),
  ],
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 100 },
}));
body.push(new Paragraph({
  children: [
    new TextRun({ text: "Keywords", font: FONT, size: 18, bold: true, italics: true }),
    new TextRun({ text: "—multi-touch attribution, Markov chain, removal effect, Shapley value, cooperative game theory, marketing budget allocation, customer journey, reproducibility", font: FONT, size: 18, bold: true }),
  ],
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 },
}));

// ================================================================== I. INTRODUCTION
H1("I", "Introduction");
P("Online advertising reaches customers through many channels at once: paid search, social media, e-mail, display advertising, referral links and direct visits. A purchase is therefore rarely the result of a single interaction. A customer may first see a display advertisement, later click a search result and finally convert after an e-mail reminder. Advertisers who must decide how to divide a marketing budget need to know how much each of these touchpoints contributed to the conversions they observe. This is the multi-touch attribution (MTA) problem [1], [2].");
P("The attribution rules most widely used in practice are positional. First-touch attribution gives all credit to the first channel in a journey, last-touch gives it all to the final channel and linear attribution divides it equally. These rules are easy to explain and compute, but they encode an assumption about which position matters instead of estimating contribution from data. A channel that frequently opens journeys that later convert through another channel receives no credit under last-touch attribution, regardless of how often its presence coincides with conversion.");
P("Two families of data-driven methods address this weakness. Graph-based or Markov-chain attribution represents journeys as transitions between channel states and measures a channel's contribution by its *removal effect*, the relative fall in the modelled conversion probability when the channel is removed from the transition graph [3]. Removal effects have recently been given a formal probabilistic treatment [4], and time-to-event formulations show that the timing of touches can also shape credit [5]. Shapley-value attribution instead treats channels as players in a cooperative game and assigns each channel its average marginal contribution over all coalitions of channels [6], [7]. Both families have been compared with rule-based models [8], and an ensemble of Markov-chain and Shapley attribution has already been proposed for budget allocation [9]. Combining the two methods is therefore *not* a new idea, and we do not claim it as one.");
P("This paper addresses a narrower and more practical question: how to turn a raw advertising interaction log into an attribution-based budget split through a pipeline in which every decision is explicit, tested and traceable, when the log contains no spend, cost or revenue. In such a setting the most a system can honestly offer is a contribution-based allocation baseline, and its value depends on whether each intermediate step, including journey reconstruction, model construction, normalization and rounding, can be audited. We study this question on a public e-commerce attribution dataset with 10,000 events and six channels.");
P("The study is organized around five research questions:", { noIndent: false });
BUL("**RQ1:** How does Markov removal-effect attribution differ from Shapley-value attribution when both are applied to the same reconstructed journeys?");
BUL("**RQ2:** Does combining normalized Markov and Shapley scores produce a more stable channel ranking than either method alone, under the evaluated weighting configurations?");
BUL("**RQ3:** How sensitive are reconstructed journeys and attribution results to the attribution window?");
BUL("**RQ4:** Can attribution scores support a transparent contribution-based budget allocation when cost and revenue are unavailable?");
BUL("**RQ5:** How do single-touch journeys and dataset-specific conversion semantics affect the interpretation of multi-touch attribution results?");
P("The contributions of this work are the following:");
BUL("An audited journey-reconstruction procedure with explicit, tested policies for row accounting, timestamp ties, conversion-delimited journeys and attribution windows that never cross an earlier conversion.");
BUL("A dual attribution design in which the Shapley value function reuses the fitted Markov model and its removal policy, together with an analytical result showing that the removal effect is a normalized leave-one-out marginal of the same game whose average marginal is the Shapley value.");
BUL("Explicit normalization and edge-case policies, including validation-gated integration of the Shapley results, a consistency check between the two engines and a positive-part policy for possible negative Shapley values.");
BUL("A transparent allocation rule that converts hybrid scores into whole-rupee amounts that reconcile exactly with the requested budget.");
BUL("A modular implementation of twelve pipeline stages covered by 473 automated tests, and a reproducible protocol for comparing attribution methods, attribution windows and hybrid weights.");
P("Section II reviews related work. Section III describes the dataset and formalizes the problem. Section IV presents the methodology, and Section V the implementation. Section VI defines the experimental protocol and Section VII reports the results. Sections VIII to XI discuss the findings, limitations, future work and conclusions.");

// ================================================================== II. RELATED WORK
H1("II", "Related Work");
H2("A", "Rule-Based and Early Data-Driven Attribution");
P("Shao and Li [1] were among the first to argue that attribution should be learned from data rather than fixed by position. They proposed a bagged logistic-regression model and a simpler probabilistic model estimated from user-level conversion data. Dalessandro *et al.* [2] argued that attribution should be causally motivated and related the credit-assignment problem to cooperative game theory. These studies established that positional heuristics are a convention, not an estimate, which is also the motivation for the present work.");
H2("B", "Markov-Chain Attribution and Removal Effects");
P("Anderl *et al.* [3] modelled customer journeys as Markov chains over channel states and attributed conversions through removal effects, studying how the order of the chain affects the results. Tao *et al.* [4] proposed a graphical point-process framework that captures the relational structure among touchpoints and defines removal effects probabilistically at the touchpoint and channel levels, and evaluated it with simulations and a real-world application. Shender *et al.* [5] described an attribution system that models conversions as an inhomogeneous Poisson process and assigns credit by iteratively removing the last advertisement in the path, motivated by advertisers who use attribution to drive budget and bidding decisions. Their emphasis on time-varying advertising effects is one motivation for our attribution-window analysis.");
H2("C", "Shapley-Value Attribution");
P("The Shapley value [6] is the unique allocation rule that satisfies efficiency, symmetry, the null-player property and additivity. Li *et al.* [7] survey its use for feature attribution in explainable artificial intelligence, including its theoretical properties, its exponential computational cost and approximation methods. In marketing, Chornous *et al.* [8] compared rule-based attribution with Markov-chain and Shapley approaches and implemented a Shapley model empirically.");
H2("D", "Hybrid and Ensemble Attribution");
P("Gaur *et al.* [9] built an ensemble of Markov-chain and Shapley attribution on transactional data from an Indian car-insurance aggregator with more than three million visitors, compared it with heuristic models, analyzed desktop and mobile channels separately and framed the result as an aid to budget allocation. This study is the closest prior work to ours and is the main reason we do not present the Markov-Shapley combination as novel.");
H2("E", "Budget Allocation");
P("Dorokhova *et al.* [10] applied computer modelling to support planning decisions for digital marketing budgets. Ben Mrad and Hnich [11] proposed an intelligent, Bayesian-network-based attribution approach aimed at improving digital marketing performance. Cai *et al.* [12] formulated marketing budget allocation as offline constrained deep reinforcement learning. Methods of this kind need cost, reward or user-level outcome information to optimize allocation. The dataset used here has none of these, which is why our allocation rule is deliberately limited to a contribution-proportional baseline.");
P("*Note on sources:* our summaries of [4], [5] and [7] to [12] are based on the published abstracts and bibliographic metadata available to us. We describe their aims and methods only and do not report their numerical findings. {{[Authors: confirm against the full texts before submission.]}}");
H2("F", "Research Gap");
P("Existing work has studied Markov-based attribution, Shapley-based attribution, removal effects and budget allocation separately, and prior research has also explored Markov-Shapley ensembles. Within the literature reviewed for this project, we did not find a study that presents the same end-to-end combination of audited journey reconstruction, explicit normalization and edge-case policies, attribution-window sensitivity, dual attribution and a transparent contribution-based budget rule designed for data without spend, cost or revenue. This is a gap observed within the reviewed literature, not proof that no such work exists. No individual component is claimed as novel. The contribution lies in the documented integration, its testing and its stated scope.");

// ================================================================== III. DATASET
H1("III", "Dataset and Problem Formulation");
H2("A", "Data Source and Schema");
P(`The input is a public multi-touch attribution event log distributed on Kaggle ${PH("dataset title, author and URL")}. It contains 10,000 rows and five columns (Table I). Timestamps have minute resolution and no time zone, and the log covers 10 February 2025 00:00 to 11 February 2025 23:59, a 48-hour window. The rows belong to 2,847 distinct users. The *Conversion* column is "Yes" in 4,944 rows and "No" in 5,056 rows. There are six channels: Direct Traffic (1,721 events), Display Ads (1,669), Email (1,654), Referral (1,685), Search Ads (1,609) and Social Media (1,662). All figures in this section are measured from the raw file by the Stage 2 validation report.`);
TABLE("Raw Schema and Its Treatment", ["Column", "Raw format", "Treatment", "Used by attribution"], [
  ["User ID", "integer as text", "kept as text key", "journey grouping"],
  ["Timestamp", "DD-MM-YYYY HH:MM", "strict parse, naive UTC", "ordering, window"],
  ["Channel", "6 labels", "trimmed; reserved names rejected", "yes (states / players)"],
  ["Campaign", "5 labels or \"-\"", "\"-\" mapped to \"(none)\"", "no (retained only)"],
  ["Conversion", "Yes / No", "mapped to 1 / 0", "journey termination"],
], [930, 1180, 1660, 1270], { note: "No event identifier, spend, cost, revenue, profit, ROI or ROAS column exists. An event identifier is derived from the original file row (Section IV-A)." });
H2("B", "Campaign Handling");
P("The *Campaign* column has five named values (New Product Launch, Retargeting, Brand Awareness, Winter Sale and Discount Offer) and the placeholder \"-\" in 3,131 rows, which preprocessing maps to the explicit label \"(none)\". Campaigns are retained in the cleaned data but are not used as attribution units. The Markov states and Shapley players are the six channel labels only. Treating campaigns as channels would mix two different levels of the marketing hierarchy and would multiply the number of Shapley players.");
H2("C", "Problem Formulation");
P("Let *N* = {1, …, *n*} be the set of marketing channels (here *n* = 6) and let each event be a tuple (*u*, *t*, *c*, *y*) of user, timestamp, channel and conversion flag. A *journey* is an ordered sequence of one user's events written as Start → *c*_{1} → … → *c*_{k} → *T*, where the terminal state *T* is Conversion or Null. Start, Conversion and Null are structural states, never channels. Given the multiset of journeys, the goal is to compute a contribution vector *a* ∈ [0, 1]^{n} with Σ_{i} *a*_{i} = 1 and, for a user-supplied budget *B* (whole rupees), a non-negative integer allocation *B*_{i} with Σ_{i} *B*_{i} = *B*.");
H2("D", "Assumptions");
P("The following assumptions are made explicitly because the dataset documentation does not settle them:");
BUL("**A1 (conversion semantics).** Every row with Conversion = Yes is a distinct conversion that happens at that touch. Its channel is the last touch of the journey, and the journey closes there. This assumption requires validation against the dataset's provenance.");
BUL("**A2 (time).** Timestamps are naive UTC with minute resolution.");
BUL("**A3 (ties).** Events of one user that share a minute are ordered by original file row. This is a deterministic convention, not a claim about true order.");
BUL("**A4 (segments).** A journey is *returning* if the same user converted earlier *in this dataset*. With 48 hours of data this is not lifetime customer status.");
P("Assumption A1 has visible consequences. Because a journey ends at the first conversion and the next event starts a new journey, a user can contribute at most one Null journey (the events after the last conversion). As a result, 77.5% of reconstructed journeys convert. This share is a property of the dataset combined with the journey definition, not a general visitor conversion rate, and it is unusually high. We therefore treat the meaning of the conversion flag as an open data-quality question (Section IX).");

// ================================================================== IV. METHODOLOGY
H1("IV", "Proposed Methodology");
H2("A", "Data Validation and Preprocessing");
P("Stage 2 follows a *no-silent-drop* policy. The file is read with every value kept as text, so that interpretation is explicit, and a 1-based row number is attached. The stage then (i) checks that the required columns exist and raises a schema error otherwise; (ii) parses timestamps strictly with the declared format; (iii) maps the conversion flag from a fixed vocabulary (yes/1/true, no/0/false); (iv) rejects rows with a missing user, an invalid timestamp, an empty channel, a channel named Start, Conversion or Null, or an invalid flag, and records the reason for each rejection; (v) handles repeated event identifiers by keeping the first exact duplicate and rejecting every row of a conflicting group; (vi) *flags but keeps* rows of the same user that share a timestamp and rows with identical content at the same time; and (vii) sorts events by (user, timestamp, event identifier) with a stable merge sort. Because the file has no event identifier, one is derived from the row number, zero-padded (r00000001, …) so that text order equals file order. This makes the tie-break of assumption A3 explicit and reproducible. The stage reports an accounting identity, rows read = rows clean + rows rejected, together with every check count, including checks that found nothing.");
H2("B", "Journey Reconstruction");
P("Stage 3 reconstructs journeys without looking ahead in time. For each event *e* of user *u*, let *y*_{e} be its flag and let *π*(*e*) be the number of conversions of *u* at strictly earlier sorted positions, computed as a per-user cumulative sum of *y* minus *y*_{e}. The event belongs to journey number *π*(*e*) + 1 of user *u*. A conversion event therefore closes its own journey, and the next event opens a new one. The path of a journey is built by wrapping its eligible channel sequence in Start and the terminal state, which is Conversion if the journey's last event is a conversion and Null otherwise. The segment label (new or returning) is computed from *π* alone, so a journey's own outcome cannot leak into it. Two accounting checks are enforced: events = touches in paths + touches excluded by the window, and converting journeys = conversion events. Algorithm 1 summarizes the procedure.");
ALGO("Algorithm 1: Journey reconstruction (Stage 3)", [
  "Input: validated events E, window w (hours or None)",
  "sort E by (user, timestamp, event_id)  # stable",
  "per user: prior <- cumsum(y) - y  # past only",
  "journey_id <- u # (prior + 1)",
  "end(j) <- max timestamp in journey j",
  "eligible(e) <- (w is None) or end(j) - t_e <= w",
  "for each journey j:",
  "   ch <- channels of eligible events, in order",
  "   T  <- Conversion if last event converts else Null",
  "   path(j) <- Start > ch > T",
  "   segment(j) <- returning if prior > 0 else new",
  "assert events == in_paths + excluded",
  "assert converting journeys == conversion events",
]);
H2("C", "Attribution Window")
P("An optional window of *w* hours limits which touches may be credited. A touch is eligible if it occurred at most *w* hours before the *final event of its journey* (inclusive boundary). The final event is always eligible, so no journey becomes empty. Journey membership is assigned before the window is applied, so an earlier conversion is a hard boundary: touches before a conversion can never be credited to a later one. As a consequence the number of journeys is identical under every window, and only path lengths change. Excluded touches are counted, never discarded silently. The default configuration uses no window, and 24-hour and 6-hour windows are evaluated for sensitivity.");
H2("D", "First-Order Absorbing Markov Chain")
P("Stage 4 fits a first-order Markov chain whose states are Start, the *n* channels, Conversion and Null. Every consecutive pair of states in every path is counted as a transition; a repeated channel (A → A) is a real self-loop. With *n*(*a*, *b*) the number of observed transitions from *a* to *b*, the transition probability is the maximum-likelihood estimate");
EQ([R("P"), rb([R("a→b")]), R("="), frac([R("n"), rb([R("a,b")])], [sum([R("n"), rb([R("a,c")])], "c")])]);
P("with no smoothing. Conversion and Null are made absorbing (*P*(*T*, *T*) = 1). A non-terminal state with no departures raises an error instead of receiving an invented distribution, and states with fewer than 10 departures are flagged as sparse. Ordering the transient states first gives the canonical form with block *Q* (transient to transient) and block *R* (transient to absorbing) [13]. The probability *h*(*i*) of eventually reaching Conversion from state *i* is the Conversion column of the absorption matrix *B* = (*I* − *Q*)^{−1}*R*:", { noIndent: true });
EQ([R("h="), sup([rb([R("I−Q")])], "−1"), R("r,  "), sub("r", "i"), R("=P"), rb([R("i→Conversion")])]);
P("The implementation solves the linear system (*I* − *Q*)*h* = *r* directly rather than forming the inverse, raises an error if the system is singular (a closed loop that can never be absorbed) and rejects solutions outside [0, 1]. A step-by-step forward iteration of the chain is used in the tests as an independent check. The baseline conversion probability is CR = *h*(Start). For a first-order chain fitted to complete journeys, CR reproduces the observed share of converting journeys exactly. The implementation verifies this identity as a sanity check of the code; it is not evidence that the model is correct. Separate models are fitted for all journeys and, if each has at least 200 journeys and 30 conversions, for the new and returning segments.", { noIndent: true });
H2("E", "Removal Effect")
P("To remove channel *k*, Stage 5 applies a *redirect-to-Null* policy. For every state *i*, the probability of moving to *k* is added to the probability of moving to Null:");
EQ([sub("P", "−k"), rb([R("i,Null")]), R("=P"), rb([R("i,Null")]), R("+P"), rb([R("i,k")]), R(",   "), sub("P", "−k"), rb([R("i,k")]), R("=0")]);
P("and *k*'s own row becomes \"go to Null\". Every row still sums to one, so *no renormalization is needed or performed*. The state *k* remains in the matrix as an unreachable dummy, which keeps the row and column labels of all models identical. The removal effect is", { noIndent: true });
EQ([sub("RE", "k"), R("=1−"), frac([sub("CR", "−k")], [R("CR")])]);
P("Under this policy CR_{−k} ≤ CR, so 0 ≤ RE_{k} ≤ 1. A value outside this range can only come from an inconsistent matrix, and the implementation raises an error rather than reporting it. The edge cases are explicit. If CR = 0, every RE is set to 0 with status *zero_baseline*, and the hybrid stage refuses to use such rows. A channel absent from the journeys receives RE = 0 with status *not_in_journeys*. If removing *k* disconnects every conversion path, then CR_{−k} = 0 and RE_{k} = 1. Removal effects measure separate counterfactual removals and do not sum to one in general.", { noIndent: true });
P("*Design versus implementation.* An earlier project specification described removal by deleting the channel and renormalizing rows, stopping with an error on a zero baseline, and allowing negative removal effects. The executable code differs on all three points, as described above. We report the implemented behaviour, because the alternative (renormalizing after deletion) silently assumes that users who would have touched *k* switch to other channels, which understates every removal effect.");
H2("F", "Shapley Value Attribution")
P("Stages 6 to 8 treat the *n* channels as players. Start, Conversion and Null are excluded by construction. Coalitions are unordered sets, and all 2^{n} = 64 are enumerated in canonical order (by size, then by name). The coalition value is defined on the fitted Markov model:");
EQ([R("v"), rb([R("S")]), R("="), sub("CR", [R("−"), rb([R("N∖S")])])]);
P("that is, the probability of eventually reaching Conversion from Start after every channel *outside* S has been removed with the redirect-to-Null policy of (3). Successive removals compose, because each only moves a column into Null. Hence *v*(*N*) = CR, and *v*(∅) is the probability of converting with no channel at all. In this dataset every journey has at least one touch, so *P*(Start → Conversion) = 0 and *v*(∅) = 0 structurally. Although a coalition is a set, *v* is evaluated on a sequence-aware chain, so channel order enters the game through the transition structure rather than through the coalition. The Shapley value of channel *i* is", { noIndent: true });
EQ([sub("φ", "i"), R("="), sum([frac([R("|S|!"), rb([R("n−|S|−1")]), R("!")], [R("n!")]), sb([R("v"), rb([R("S∪{i}")]), R("−v"), rb([R("S")])])], "S⊆N∖{i}")]);
P("The weight is the probability that exactly the members of S precede *i* in a uniformly random ordering of the channels, so φ_{i} is *i*'s average marginal contribution over all orderings [6]. The implementation enumerates all *n*·2^{n−1} = 192 marginal terms exactly, sums them with compensated summation (math.fsum) in a fixed order, raises an error if any coalition value is missing, and refuses to run with more than 20 channels. Efficiency is checked to a tolerance of 10^{−9}:", { noIndent: true });
EQ([sum([sub("φ", "i")], "i∈N"), R("=v"), rb([R("N")]), R("−v"), rb([R("∅")])]);
P("*Sign of the Shapley values.* Redirecting probability mass to Null can never increase the probability of absorption in Conversion, so *v*(S) ≤ *v*(S ∪ {*i*}) for every S and *i*. This is the same property that bounds RE_{k} in [0, 1]. The game is therefore monotone and every φ_{i} ≥ 0 under this value function. Negative Shapley values remain possible for other value functions, which is why the downstream stages still handle them explicitly (Section IV-H).", { noIndent: true });
P("Stage 9 validates the Shapley outputs with 13 checks before Stage 10 is allowed to integrate them: coalition completeness, validity of the values as probabilities, one finite value per channel with no missing, duplicate or unexpected channels, efficiency, an independent reference computation that averages marginals over all *n*! orderings and never calls the production code, synthetic games with hand-derived answers, symmetry and null-player properties, and determinism under row and channel reordering. A failed check blocks integration and nothing is written.");
H2("G", "Relationship Between the Two Scores")
P("Because both scores are computed from the same game, (3), (4) and (5) give");
EQ([sub("RE", "i"), R("="), frac([R("v"), rb([R("N")]), R("−v"), rb([R("N∖{i}")])], [R("v"), rb([R("N")])])]);
P("The removal effect is therefore the marginal contribution of *i* when it joins *last*, scaled by *v*(*N*), while φ_{i} averages the marginal contribution of *i* over every joining position. If *v* were additive, every marginal of *i* would be equal and the two normalized scores would coincide. Differences between them thus measure non-additivity (synergy or redundancy) in the Markov game, not disagreement between two independent models. This framing matters when interpreting the hybrid: it interpolates between two summaries of one model, so agreement between the methods is partly built in and should not be read as independent corroboration.", { noIndent: true });
H2("H", "Score Normalization")
P("The Markov weights are");
EQ([sub("M̂", "i"), R("="), frac([sub("RE", "i")], [sum([sub("RE", "j")], "j")])]);
P("which require every RE to be finite and non-negative and the total to exceed 10^{−12}; otherwise the hybrid stage raises an error. Raw Shapley values are kept unchanged for reporting. Stage 10 adds a *signed* weight φ_{i}/Σφ with a status flag (OK, signed values present, or undefined for a non-positive total). For allocation, Stage 11 applies a *positive-part* policy:", { noIndent: true });
EQ([sub("φ̂", "i"), R("="), frac([R("max"), rb([sub("φ", "i"), R(",0")])], [sum([R("max"), rb([sub("φ", "j"), R(",0")])], "j")])]);
P("Clipped channels are flagged. If every value clips to zero the stage raises an error, and no equal-split fallback is used. Clipping is a policy decision, not a property of Shapley values, because it discards how negative a channel was. Under the monotone game of Section IV-F it is never triggered, and (10) reduces to φ_{i}/*v*(*N*).", { noIndent: true });
H2("I", "Hybrid Attribution")
P("The hybrid score of channel *i* is");
EQ([sub("H", "i"), R("=α"), sub("M̂", "i"), R("+"), rb([R("1−α")]), sub("φ̂", "i"), R(",   α∈"), sb([R("0,1")])]);
P("Because both weight vectors are non-negative and sum to one, the H_{i} also form a distribution, and the implementation checks this to 10^{−9} rather than renormalizing. Channels are matched by name, never by position, and any mismatch, duplicate, reserved state name, non-finite value or non-*ok* Markov status raises an error. Before combining the scores, a consistency guard checks that Σφ_{i} equals the Markov baseline CR within 10^{−5}, which is required by (7) when *v*(∅) = 0. If the guard fails, the two engines were run on different journeys or definitions and the stage refuses to proceed. Since the Shapley stage has no segment dimension, only the all-journeys model is combined. The default α = 0.5 is a neutral starting point, *not* an optimized value. The weight is a command-line parameter, and Section VI defines a sweep over it.", { noIndent: true });
H2("J", "Contribution-Based Budget Allocation")
P("Given a whole-rupee budget *B* and hybrid weights *w*_{i} = H_{i}/Σ_{j}H_{j}, the exact share of channel *i* is");
EQ([sub("q", "i"), R("=B·"), sub("w", "i")]);
P("Stage 12 computes the shares with exact rational arithmetic, floors them to whole rupees and gives the remaining *L* = *B* − Σ⌊*q*_{i}⌋ rupees (always fewer than *n*) one each to the channels with the largest fractional parts, breaking ties alphabetically. This is the largest-remainder (Hamilton) apportionment method [14]. As a result the amounts sum to *B* exactly, each is within one rupee of its exact share, a zero-weight channel never receives a leftover rupee and the output does not depend on input row order. The budget must be a finite whole number between 0 and 10^{15}, and fractional budgets are rejected rather than rounded. The weights must be finite, non-negative, unique per channel and free of structural state names, and they must sum to 1 within 10^{−6}, so an all-zero weight vector is rejected with an error. Minimum or maximum channel budgets are *not* implemented.", { noIndent: true });
P("*Illustrative example (not a result).* With hypothetical weights of 0.40, 0.30, 0.20 and 0.10 for four channels A to D and *B* = ₹1,00,000, the allocation is ₹40,000, ₹30,000, ₹20,000 and ₹10,000. Because no spend or revenue is observed, a larger allocation means a larger *attributed contribution*. It does not mean a higher return, and the rule is not a financial optimizer.");

// ================================================================== V. ARCHITECTURE
H1("V", "System Architecture and Implementation");
H2("A", "Overview");
P("The system is an offline Python prototype organized as twelve stages, each with a defined input file, output file and test module (Fig. 1, Table II). The Markov and Shapley engines read the same journeys file and meet at the hybrid stage. The Shapley value function reuses the Markov engine's fitted matrix and removal function rather than reimplementing them, so a change to the removal policy propagates to both scores. All intermediate outputs are CSV files written at full floating-point precision, with deterministic row order and tie-breaking.");
FIG(path.join(BUILD, "fig1_architecture.png"), "Implemented pipeline. Both attribution engines consume the same reconstructed journeys; the Shapley value function V(S) is evaluated on the Markov engine's fitted transition matrix using the same redirect-to-Null removal policy. The Streamlit interface is not yet implemented.", 300);
TABLE("Modules, Interfaces and Tests", ["Stage / module", "Output", "Tests"], [
  ["2  data_pipeline", "validated events + report", "25"],
  ["3  journey_builder", "journeys.csv", "30"],
  ["4  markov_engine (fit)", "count / probability matrices", "37"],
  ["5  markov_engine (removal)", "markov_results.csv", "28"],
  ["6  coalition_generator", "coalitions.csv", "39"],
  ["7  coalition_value", "coalition_values.csv", "33"],
  ["8  shapley_engine", "shapley_values.csv", "28"],
  ["9  shapley_validation", "13-check report", "67"],
  ["10 shapley_integration", "final_shapley_attribution.csv", "34"],
  ["11 hybrid_engine", "hybrid_attribution.csv", "70"],
  ["12 budget_allocator", "allocation table", "82"],
  ["Total", "", "473"],
], [1900, 2340, 800], { note: "Each stage reads the previous stage's file and raises an error on missing columns, reserved state names or invalid values. Test counts are measured from the repository's test collection." });
H2("B", "Software Environment and Reproducibility");
P(`The measurements in this paper were produced with Python 3.14.7, pandas 3.0.6, NumPy 2.5.4 and pytest 9.1.1 on Windows 11. Each stage is run from the project root with a script (scripts/run_stage3.py to run_stage12.py). The hybrid weight and the budget are command-line options (for example, run_stage11.py --alpha 0.5 and run_stage12.py --budget 10,00,000). No step is stochastic, so no random seed is needed. The dependency file lists packages without versions. Pinning them is a necessary step for strict reproducibility ${PH("repository URL and commit hash used for the reported results")}.`);
H2("C", "Testing");
P("The full suite of 473 tests passes in about 14 s (measured). Most tests use small synthetic inputs with hand-derivable answers, for example a nine-journey worked example whose baseline conversion probability is 7/9. These tests cover malformed inputs, ties, windows, absorbing-state validity, removal edge cases, coalition completeness, Shapley axioms, normalization failures and exact budget reconciliation. Further tests run on the real data file and on the stage outputs when these are present. In the reported run all files were present and no test was skipped. Passing tests show that the implementation agrees with its specification; they do not show that the specification captures true channel contribution.");
H2("D", "Interface Status");
P("A Streamlit front end is planned to let users choose α and the budget and to display the allocation together with a fixed interpretation notice that the code already defines (no ROI, no causal claim). It has not been implemented, and this paper makes no claim about it.");

// ================================================================== VI. EXPERIMENTAL SETUP
H1("VI", "Experimental Setup");
H2("A", "Configurations");
P("All experiments use the same validated events and the journey definition of Section IV-B. The factors studied are the attribution window (none, 24 h and 6 h), the segment (all, new and returning, for the Markov engine), the hybrid weight α ∈ {0.0, 0.1, …, 1.0} with the Shapley weight 1 − α, and the budget *B* (the command-line default is ₹10,00,000). Varying the window requires regenerating the journeys file and rerunning Stages 4 to 12 for each setting.");
H2("B", "Baselines");
P("First-touch, last-touch and linear attribution serve as heuristic reference points, not as ground truth. They are computed on *converting* journeys only. First-touch gives one unit of credit to the first eligible touch, last-touch to the last eligible touch (here the touch that carries the conversion, by assumption A1), and linear gives 1/*k* to each of the *k* eligible touches, so a channel that appears twice in a journey receives 2/*k*. Credits are summed per channel and normalized to shares. The Markov and Shapley engines also use non-converting journeys, so baseline shares are compared through normalized shares and rankings and never through raw magnitudes. {{[Status: the baseline module is not yet in the repository; implement and test it before reporting Table IX.]}}");
H2("C", "Metrics");
P("For each method we report the normalized share and the rank of every channel. Agreement between two methods is measured with Spearman's rank correlation");
EQ([R("ρ=1−"), frac([R("6"), sum([sup([sub("d", "i")], "2")], "i")], [R("n"), rb([sup("n", "2"), R("−1")])])]);
P("where *d*_{i} is the rank difference of channel *i*, and with Kendall's τ, which counts concordant and discordant channel pairs. We also report the largest absolute share difference and the L1 distance between share vectors. Stability across windows and across α is summarized by the same statistics computed against a reference configuration. With *n* = 6 channels, rank correlations take few distinct values and are coarse, so they are reported together with the share differences. Rank agreement shows that two methods order channels similarly. It does not show that either ordering is correct. For computational cost we report the number of channels, the number of coalitions evaluated, wall-clock runtime per stage and the hardware used.", { noIndent: true });
H2("D", "Research Questions and Evidence");
TABLE("Research Questions, Analyses and Evidence Status", ["RQ", "Analysis", "Evidence status"], [
  ["RQ1", "RE vs. φ shares, ranks, ρ, τ (Tables VI–VIII); Eq. (8)", "analytical link derived; numbers " + "to be inserted"],
  ["RQ2", "α sweep; rank stability of H vs. M̂ and φ̂", "to be executed"],
  ["RQ3", "Windows: reconstruction (Table V) and attribution", "reconstruction measured; attribution to be executed"],
  ["RQ4", "Stage 12 on hybrid weights; reconciliation checks", "rule implemented and tested; allocation to be inserted"],
  ["RQ5", "Single-touch share, conversion share, A1", "measured; interpretation in Sec. VIII"],
], [500, 2540, 2000], { leftAll: true });

// ================================================================== VII. RESULTS
H1("VII", "Results");
P("Each result below is labelled as *measured* (produced directly by executing the pipeline), *derived* (computed from measured values with a stated formula) or *to be inserted*. Highlighted fields are placeholders for results that will be added from verified runs.", { noIndent: false });
H2("A", "Validation and Journey Reconstruction (Measured)");
P("All 10,000 rows passed validation, and no row was rejected. Ten rows (five pairs of events) share a user and minute; they were kept, flagged and ordered by file row. No identical-content duplicates were found. Table IV summarizes the reconstruction without a window, and both accounting identities hold.");
TABLE("Journey Reconstruction (No Window)", ["Quantity", "Value", "Type"], [
  ["Events / users", "10,000 / 2,847", "measured"],
  ["Rows rejected", "0", "measured"],
  ["Journeys", "6,382", "measured"],
  ["Converting / Null journeys", "4,944 / 1,438", "measured"],
  ["Converting share", "77.5% (4,944/6,382)", "derived"],
  ["New / returning journeys", "2,847 / 3,535", "measured"],
  ["Conversions: new / returning", "2,381 / 2,563", "measured"],
  ["Single-touch journeys", "3,926 (61.5%)", "measured / derived"],
  ["Touches per journey: mean / max", "1.57 / 8", "measured"],
  ["Journeys with a same-minute tie", "8 (6 converting)", "measured"],
  ["Repeated channel touches by a user", "2,360", "measured"],
], [2440, 1700, 900]);
P("Every user has exactly one *new* journey (2,847), which matches the definition: a user's first journey cannot be preceded by an observed conversion. Each user contributes at most one Null journey, and 1,438 users have trailing events after their last conversion or never convert. In six converting journeys the tie-break decides which channel carries the conversion. This affects at most six of 4,944 terminal channels, but it is an arbitrary choice and is disclosed as such.");
H2("B", "Attribution-Window Sensitivity of Journeys (Measured)");
TABLE("Effect of the Attribution Window on Reconstructed Journeys", ["Window", "Touches kept", "Excluded", "Mean", "Max", "Single-touch"], [
  ["None", "10,000", "0 (0.0%)", "1.57", "8", "3,926"],
  ["24 h", "9,444", "556 (5.6%)", "1.48", "7", "4,109"],
  ["6 h", "7,539", "2,461 (24.6%)", "1.18", "4", "5,319"],
], [700, 900, 1100, 600, 500, 1240], { note: "Journey count (6,382) and the number of converting journeys (4,944) are identical under all windows by construction (Section IV-C). Percentages are derived from the measured counts." });
P("Shortening the window removes older touches, shortens journeys and increases the number of single-touch journeys from 3,926 to 5,319 (Fig. 2). Under the 6-hour window, 83.3% of journeys (derived) contain a single touch, which leaves little sequential information for either attribution engine. We do not treat any window as empirically best. The choice should follow the business's purchase-cycle knowledge, together with the attribution-level stability analysis of Section VII-F.");
FIG(path.join(BUILD, "fig2_window.png"), "Touches retained and excluded, and mean touches per journey, under no window, a 24-hour window and a 6-hour window (measured, Stage 3).", 318);
H2("C", "Markov Removal Effects");
P(`Without a window, the fitted chain has nine states (Start, six channels, Conversion and Null). The baseline conversion probability equals the converting share, CR = 4,944/6,382 ≈ 0.7747 (derived), as the identity in Section IV-D requires. ${PH("sparse-state flags, sum of removal effects, and a description of the ranking; segment models (new, returning) if reported")}`);
TABLE("Markov Removal Effects (All Journeys, No Window)", ["Channel", "CR without", "RE", "M̂ (%)", "Rank"],
  CH.map((c) => [c, "{{ — }}", "{{ — }}", "{{ — }}", "{{ — }}"]), [1300, 950, 900, 1000, 890]);
H2("D", "Shapley Values");
P(`The game has 6 players and 64 coalitions (derived), with *v*(∅) = 0 and *v*(*N*) = CR. ${PH("efficiency gap, outcome of the 13 validation checks, whether any value was clipped, and the ranking")}`);
TABLE("Shapley Attribution (All Journeys, No Window)", ["Channel", "φ (raw)", "φ̂ (%)", "Rank"],
  CH.map((c) => [c, "{{ — }}", "{{ — }}", "{{ — }}"]), [1500, 1300, 1300, 940]);
H2("E", "Hybrid Attribution and Weight Sensitivity");
TABLE("Hybrid Scores at α = 0.5 and Rank Agreement", ["Channel", "M̂ (%)", "φ̂ (%)", "H (%)", "Rank"],
  CH.map((c) => [c, "{{ — }}", "{{ — }}", "{{ — }}", "{{ — }}"]), [1300, 950, 950, 950, 890],
  { note: `Spearman ρ and Kendall τ between the M̂ and φ̂ rankings: ${PH("ρ, τ")}.` });
FIG_PLACEHOLDER("Hybrid share of each channel as α varies from 0 (Shapley only) to 1 (Markov only).", "α-sweep line chart, one line per channel, from run_stage11.py --alpha 0.0 … 1.0");
P(PH("describe how the ranking changes across α, and identify the α values at which ranks swap"));
H2("F", "Attribution-Window Sensitivity of Scores");
P(PH("hybrid (and, optionally, M̂ and φ̂) shares and ranks under the 24 h and 6 h windows, with ρ and τ against the no-window ranking"));
H2("G", "Comparison With Heuristic Baselines");
TABLE("Channel Shares (%) by Attribution Method", ["Channel", "First", "Last", "Linear", "Markov", "Shapley", "Hybrid"],
  CH.map((c) => [c, "{{ — }}", "{{ — }}", "{{ — }}", "{{ — }}", "{{ — }}", "{{ — }}"]), [1140, 650, 650, 650, 650, 650, 650],
  { note: "Heuristic shares are computed on converting journeys only (Section VI-B)." });
H2("H", "Budget Allocation");
TABLE(`Contribution-Based Allocation of ₹10,00,000 (α = 0.5)`, ["Channel", "H (%)", "Allocation (₹)", "Rounding adj. (₹)"],
  [...CH.map((c) => [c, "{{ — }}", "{{ — }}", "{{ — }}"]), ["Total", "100", "10,00,000", "0"]], [1400, 1000, 1500, 1140],
  { note: "Amounts are whole rupees, and the total reconciles exactly by construction (Section IV-J). These amounts are hypothetical and contribution-based; they are not ROI-optimized." });
H2("I", "Computational Cost");
TABLE("Computational Profile", ["Quantity", "Value", "Type"], [
  ["Channels (players) n", "6", "measured"],
  ["Coalitions 2^{n}", "64", "derived"],
  ["Marginal terms n·2^{n−1}", "192", "derived"],
  ["Markov fit + removal effects (s)", "{{ — }}", "measured"],
  ["Coalition values + Shapley (s)", "{{ — }}", "measured"],
  ["Hardware", PH("CPU, RAM"), "—"],
], [2440, 1700, 900]);
P("Each coalition value requires one linear solve over at most eight transient states, so the cost is dominated by the 2^{n} growth in the number of coalitions rather than by the Markov computation. At the configured limit of *n* = 20 channels, exact enumeration would require 1,048,576 coalition values.");

// ================================================================== VIII. DISCUSSION
H1("VIII", "Discussion");
H2("A", "Why the Two Scores Differ");
P("Equation (8) shows that, in this pipeline, the Markov and Shapley scores are two summaries of one game. The removal effect looks only at what is lost when a channel leaves the full set of channels. The Shapley value averages what a channel adds across all subsets. They diverge when a channel's marginal contribution depends on which other channels are present, for example when two channels mostly appear in the same multi-touch journeys and partly substitute for each other. Their difference is therefore a direct diagnostic of interaction in the fitted chain. It is not a contest between independent models. " + PH("relate to the measured differences in Tables VI and VII"));
H2("B", "Sequence and Coalition Information");
P("Sequence information enters both scores only through the first-order transition probabilities. The chain does not remember how a journey reached its current state, and it does not use the time gaps between touches. Coalition information enters through the averaging in (6). Neither score uses campaign labels, and both are defined relative to the journey definition of Section IV-B.");
H2("C", "Effect of Single-Touch Journeys");
P("Because the conversion event is itself a touch (assumption A1), 61.5% of journeys have the form Start → *c* → *T*. In a single-touch converting journey every method, including first-touch, last-touch and linear attribution, gives all credit to *c*. Methods can therefore differ only through the 2,456 multi-touch journeys (derived). Together with the near-uniform channel frequencies (1,609 to 1,721 events per channel), this suggests that channel scores may lie close together, so small numerical differences should not be read as meaningful differences in contribution. " + PH("confirm with measured spread of scores; consider bootstrap intervals"));
H2("D", "Attribution Windows");
P("Shorter windows move every method toward last-touch behaviour, because more journeys are reduced to their converting touch. Window sensitivity is therefore partly a question of how much sequence information the analyst is willing to discard. It should be decided from knowledge of the purchase cycle, not by choosing the window that produces a preferred ranking.");
H2("E", "Heuristic Baselines and Practical Use of the Allocation");
P(`${PH("summarize where the data-driven methods agree or disagree with first-touch, last-touch and linear attribution")} A difference from the heuristics shows that the models assign credit differently. It does not show that they assign it more accurately, because the data contain no ground-truth contribution. The allocation in Table X is best read as a transparent default: a proportional split of a fixed budget according to modelled contribution, which can be audited back to the journey counts. It should not be read as a forecast of returns.`);

// ================================================================== IX. LIMITATIONS
H1("IX", "Limitations");
P("*Data.* The log has no spend, cost, revenue or profit, so ROI and ROAS cannot be computed and the allocation cannot be profit-maximizing. The meaning of the conversion flag is undocumented. A converting-journey share of 77.5% is unusually high, and the assumption that each flagged row is a distinct conversion closing one journey (A1) needs validation, especially for users with several conversions. The data cover only 48 hours, so long journeys and earlier purchases are invisible, and the new/returning labels are relative to the dataset. Same-minute ties are resolved by file order, which is a convention and not observed order.");
P("*Modelling.* Both scores are model-based and not causal. The first-order chain ignores path history and timing, and it uses unsmoothed frequencies. The results depend on the journey definition, the redirect-to-Null policy, the value function, the normalization policies and α. Exact Shapley computation grows as 2^{n}. A high share of single-touch journeys limits the evidence about channel interactions. A stable ranking is not evidence that the ranking is correct.");
P("*Evaluation and implementation.* There is no ground truth against which to measure accuracy, and the allocation has not been shown to improve any business outcome. The heuristic baselines, the α sweep and the attribution-level window analysis still need to be executed (Section VII). The dependencies are not pinned, and the Streamlit interface has not been built.");

// ================================================================== X. FUTURE WORK
H1("X", "Future Work");
P("On the data side, the priorities are to validate the conversion semantics with the dataset's provenance, to add channel-level spend, cost and revenue, and to use longer logs with richer user and campaign features. On the modelling side, natural extensions are higher-order or time-aware Markov models, Shapley approximations for larger channel sets, bootstrap confidence intervals for scores and ranks, and causal designs such as holdout experiments that can support incremental-lift claims. With cost data available, the proportional rule could be replaced by ROI-aware constrained optimization with channel floors and caps [12]. On the deployment side, the next steps are to complete the Streamlit interface, pin dependencies, and add incremental data ingestion.");

// ================================================================== XI. CONCLUSION
H1("XI", "Conclusion");
P("This paper presented an auditable pipeline that turns a raw advertising interaction log into journeys, Markov removal effects, Shapley values, a configurable hybrid score and an exactly reconciled, contribution-based budget split. Every stage states and tests its policies, and every row and conversion is accounted for. On a public 10,000-event log, reconstruction was complete and reproducible, but it also showed that most journeys are single-touch and that the conversion semantics need scrutiny. The analysis shows that, under the implemented value function, the removal effect and the Shapley value are leave-one-out and average marginal contributions of the same game. The hybrid is therefore a principled interpolation between them rather than a combination of independent evidence. The pipeline does not establish causality or financial optimality. Its value is in making attribution-based budget decisions transparent and traceable, and in defining what richer data would be needed to go further. " + PH("one sentence on the main verified attribution finding"));

// ================================================================== back matter
body.push(new Paragraph({ children: [new TextRun({ text: "Author Contributions", font: FONT, size: 20, smallCaps: true })], alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 } }));
P(`J. Doshi designed and implemented data validation, journey reconstruction, attribution windows, the Markov engine and removal effects. S. Gupta designed and implemented the coalition generator, the coalition value function, the Shapley engine, its validation suite and the validation-gated integration. ${PH("confirm who implemented the hybrid engine and the budget allocator, and add any further authors or the project guide")}`);
body.push(new Paragraph({ children: [new TextRun({ text: "Acknowledgment", font: FONT, size: 20, smallCaps: true })], alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 } }));
P(PH("acknowledgment of guide, department and institution"));

body.push(new Paragraph({ children: [new TextRun({ text: "References", font: FONT, size: 20, smallCaps: true })], alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 } }));
const REFS = [
  "X. Shao and L. Li, \"Data-driven multi-touch attribution models,\" in *Proc. 17th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2011, pp. 258–264, doi: 10.1145/2020408.2020453.",
  "B. Dalessandro, C. Perlich, O. Stitelman, and F. Provost, \"Causally motivated attribution for online advertising,\" in *Proc. 6th Int. Workshop Data Mining for Online Advertising and Internet Economy (ADKDD)*, 2012, pp. 1–9, doi: 10.1145/2351356.2351363.",
  "E. Anderl, I. Becker, F. von Wangenheim, and J. H. Schumann, \"Mapping the customer journey: Lessons learned from graph-based online attribution modeling,\" *Int. J. Res. Marketing*, vol. 33, no. 3, pp. 457–474, 2016, doi: 10.1016/j.ijresmar.2016.03.001.",
  "J. Tao, Q. Chen, J. W. Snyder, A. S. Kumar, A. Meisami, and L. Xue, \"A graphical point process framework for understanding removal effects in multi-touch attribution,\" *Management Science*, vol. 71, no. 9, pp. 7312–7332, 2025, doi: 10.1287/mnsc.2023.00457.",
  "D. Shender, A. N. Amini, X. Bao, M. Dikmen, J. Wang, and A. R. Fricke, \"A time to event framework for multi-touch attribution,\" *J. Data Science*, pp. 56–76, 2023, doi: 10.6339/23-JDS1101.",
  "L. S. Shapley, \"A value for n-person games,\" in *Contributions to the Theory of Games II*, H. W. Kuhn and A. W. Tucker, Eds. Princeton, NJ, USA: Princeton Univ. Press, 1953, pp. 307–317.",
  "M. Li, H. Sun, Y. Huang, and H. Chen, \"Shapley value: From cooperative game to explainable artificial intelligence,\" *Autonomous Intelligent Systems*, vol. 4, no. 1, 2024, doi: 10.1007/s43684-023-00060-8.",
  "G. Chornous, M. Petrova, M. Horna, N. Penev, and M. Pustovit, \"Data-driven attribution modeling in digital marketing,\" *Bulletin of Taras Shevchenko National University of Kyiv. Economics*, no. 228, pp. 191–201, 2026, doi: 10.17721/1728-2667.2026/228-1/20.",
  "J. Gaur, K. Bharti, and R. Bajaj, \"Maximizing marketing impact: Heuristic vs ensemble models for attribution modeling,\" *Global Knowledge, Memory and Communication*, vol. 75, no. 1–2, pp. 359–377, 2026 (published online 2024), doi: 10.1108/GKMC-04-2023-0112.",
  "L. Dorokhova, A. Kuusik, R. Dimitrov, K. Pentus, O. Dorokhov, and M. Petrova, \"Planning the digital marketing budget: Computer modelling for decision making,\" *Access J. – Access to Science, Business, Innovation in the Digital Economy*, vol. 4, no. 2, pp. 248–260, 2023, doi: 10.46656/access.2023.4.2(7).",
  "A. Ben Mrad and B. Hnich, \"Intelligent attribution modeling for enhanced digital marketing performance,\" *Intelligent Systems with Applications*, vol. 21, Art. no. 200337, 2024, doi: 10.1016/j.iswa.2024.200337.",
  "T. Cai *et al.*, \"Marketing budget allocation with offline constrained deep reinforcement learning,\" in *Proc. 16th ACM Int. Conf. Web Search and Data Mining (WSDM)*, 2023, pp. 186–194, doi: 10.1145/3539597.3570486.",
  "J. G. Kemeny and J. L. Snell, *Finite Markov Chains*. Princeton, NJ, USA: Van Nostrand, 1960.",
  "M. L. Balinski and H. P. Young, *Fair Representation: Meeting the Ideal of One Man, One Vote*. New Haven, CT, USA: Yale Univ. Press, 1982.",
];
REFS.forEach((r, i) => body.push(new Paragraph({
  children: [new TextRun({ text: `[${i + 1}]`, font: FONT, size: 16 }), new TextRun({ text: "\t", font: FONT, size: 16 }), ...runs(r, { size: 16 })],
  tabStops: [{ type: TabStopType.LEFT, position: 400 }], indent: { left: 400, hanging: 400 },
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 30 },
})));

// ================================================================== title block (single column)
const titleBlock = [
  new Paragraph({
    children: [new TextRun({ text: "Multi-Touch Attribution and Contribution-Based Budget Allocation Using Markov Chains and Shapley Values", font: FONT, size: 48 })],
    alignment: AlignmentType.CENTER, spacing: { before: 0, after: 200 },
  }),
  new Paragraph({
    children: [
      new TextRun({ text: "Jaanvi Doshi", font: FONT, size: 22 }),
      new TextRun({ text: "                    ", font: FONT, size: 22 }),
      new TextRun({ text: "Shivika Gupta", font: FONT, size: 22 }),
      new TextRun({ text: "                    ", font: FONT, size: 22 }),
      ...runs("{{[Guide / co-author]}}", { size: 22 }),
    ],
    alignment: AlignmentType.CENTER, spacing: { after: 40 },
  }),
  new Paragraph({
    children: [...runs("*Department of Computer Science and Engineering,* ", { size: 18 }), ...runs("{{[Institution, City, India]}}", { size: 18 })],
    alignment: AlignmentType.CENTER, spacing: { after: 20 },
  }),
  new Paragraph({
    children: runs("{{[e-mail addresses]}}", { size: 18 }),
    alignment: AlignmentType.CENTER, spacing: { after: 240 },
  }),
];

const page = { size: { width: 12240, height: 15840 }, margin: { top: 1080, right: 900, bottom: 1080, left: 900, footer: 500 } };
const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16 })] })] });

const doc = new Document({
  creator: "Jaanvi Doshi",
  title: "Multi-Touch Attribution and Contribution-Based Budget Allocation Using Markov Chains and Shapley Values",
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 300, hanging: 200 } } } }] }] },
  sections: [
    { properties: { page }, footers: { default: footer }, children: titleBlock },
    { properties: { page, type: SectionType.CONTINUOUS, column: { count: 2, space: 360, equalWidth: true } }, footers: { default: footer }, children: body },
  ],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, buf.length); });
