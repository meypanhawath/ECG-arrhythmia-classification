"""ECG arrhythmia classification pipeline.

Generated from the notebook, so the module and the notebook cannot drift apart.
Cells that only work inside Colab (Drive mount, GitHub push) are left out.
"""


# ============================================================================
# ECG Arrhythmia Classification
# ============================================================================


# ============================================================================
# Medical terms used in this notebook / ពាក្យបច្ចេកទេសវេជ្ជសាស្ត្រ
# ============================================================================


# ============================================================================
# 1. Setup
# ============================================================================

DATA    = "/content/drive/MyDrive/ecg_mitbih"
PROJ    = os.path.join(DATA, "project")
FIG     = os.path.join(PROJ, "results", "figures")
TAB     = os.path.join(PROJ, "results", "tables")
RES     = os.path.join(PROJ, "results")
NBDIR   = os.path.join(PROJ, "notebook")
SIG_DIR = os.path.join(DATA, "signals")
FEAT    = os.path.join(DATA, "ecg_features.parquet")

for d in (FIG, TAB, RES, NBDIR, SIG_DIR):
    os.makedirs(d, exist_ok=True)

FS, SEED, CAP = 360, 42, 25000
REBUILD_FEATURES = False

CLASSES    = ["N", "S", "V", "F"]
CLASS_NAME = {"N": "Normal", "S": "Supraventricular", "V": "Ventricular", "F": "Fusion"}
CLASS_COL  = {"N": "#2F3C7E", "S": "#C87A2A", "V": "#990011", "F": "#5F7A52"}
SPLIT_COL  = {"random": "#8FA8D8", "inter-patient": "#990011"}

AAMI = {"N": ["N","L","R","e","j"], "S": ["A","a","J","S"],
        "V": ["V","E"], "F": ["F"], "Q": ["/","f","Q"]}
SYM2AAMI  = {s: k for k, v in AAMI.items() for s in v}
BEAT_SYMS = set(SYM2AAMI)

DS1 = ["101","106","108","109","112","114","115","116","118","119","122",
       "124","201","203","205","207","208","209","215","220","223","230"]
DS2 = ["100","103","105","111","113","117","121","123","200","202","210",
       "212","213","214","219","221","222","228","231","232","233","234"]
RECORDS = DS1 + DS2
PACED   = ["102","104","107","217"]

RR_NAMES   = ["pre_rr","post_rr","local_rr","global_rr",
              "pre_over_local","pre_over_global","post_over_pre","local_over_global"]
STAT_NAMES = ["skew","kurt","energy","rpeak_amp","min_amp","qrs_width"]

RESULTS = {}

def save_table(df, name, index=False, show=True):
    df.to_csv(os.path.join(TAB, name + ".csv"), index=index)
    RESULTS[name] = df
    if show:
        print(df.to_string(index=index))
    return df

def save_fig(fig, name, show=True):
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)

print("project :", PROJ)
print("records :", len(RECORDS), "| excluded:", PACED, "| classes:", CLASSES)


# ============================================================================
# 2. Data collection
# ============================================================================

def load_signal(rec):
    cache = os.path.join(SIG_DIR, rec + ".npy")
    meta  = os.path.join(SIG_DIR, rec + "_lead.txt")
    if os.path.exists(cache):
        return np.load(cache), open(meta).read().strip()
    r = wfdb.rdrecord(rec, pn_dir="mitdb")
    ch = r.sig_name.index("MLII") if "MLII" in r.sig_name else 0
    sig = r.p_signal[:, ch].astype(np.float32)
    np.save(cache, sig)
    open(meta, "w").write(r.sig_name[ch])
    return sig, r.sig_name[ch]

t0 = time.time()
leads = {rec: load_signal(rec)[1] for rec in tqdm(RECORDS, desc="records")}
r0 = wfdb.rdrecord("100", pn_dir="mitdb")

source = pd.DataFrame([
    {"property": "Database",        "value": "MIT-BIH Arrhythmia Database"},
    {"property": "Source",          "value": "PhysioNet (open access)"},
    {"property": "Recordings used", "value": f"{len(RECORDS)} of 48"},
    {"property": "Excluded (paced)","value": ", ".join(PACED)},
    {"property": "Duration each",   "value": f"{r0.p_signal.shape[0]/FS/60:.1f} min"},
    {"property": "Sampling rate",   "value": f"{FS} Hz"},
    {"property": "Samples/record",  "value": f"{r0.p_signal.shape[0]:,}"},
    {"property": "Lead used",       "value": ", ".join(f"{k} x{v}" for k, v in pd.Series(leads).value_counts().items())},
    {"property": "Load time",       "value": f"{time.time()-t0:.0f} s"},
])
save_table(source, "01_data_source")


# ============================================================================
# 3. Data exploration
# ============================================================================

rows = []
for rec in tqdm(RECORDS, desc="annotations"):
    a = wfdb.rdann(rec, "atr", pn_dir="mitdb")
    for s, y in zip(a.sample, a.symbol):
        if y in BEAT_SYMS:
            rows.append({"record": rec, "symbol": y, "aami": SYM2AAMI[y],
                         "split": "DS1" if rec in DS1 else "DS2"})
ann = pd.DataFrame(rows)

sym = (ann["symbol"].value_counts().rename_axis("symbol").reset_index(name="beats"))
sym["aami"] = sym["symbol"].map(SYM2AAMI)
save_table(sym, "02_raw_symbols")

vc = ann[ann.aami.isin(CLASSES)]["aami"].value_counts().reindex(CLASSES)
dist = pd.DataFrame({
    "class": CLASSES,
    "meaning": [CLASS_NAME[c] for c in CLASSES],
    "beats": vc.values,
    "share_pct": (vc.values / vc.sum() * 100).round(2),
})
dist["ratio_to_N"] = (vc.max() / vc.values).round(0).astype(int)
save_table(dist, "03_class_distribution")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
ax[0].barh(dist["class"], dist["beats"], color=[CLASS_COL[c] for c in CLASSES])
for i, v in enumerate(dist["beats"]):
    ax[0].text(v, i, f" {v:,}", va="center", fontsize=9)
ax[0].set_xlabel("beats"); ax[0].set_title("class distribution")
ax[0].invert_yaxis(); ax[0].set_xlim(0, dist["beats"].max()*1.25)

ax[1].barh(dist["class"], dist["beats"], color=[CLASS_COL[c] for c in CLASSES])
ax[1].set_xscale("log"); ax[1].set_xlabel("beats (log scale)")
ax[1].set_title(f"imbalance {dist['beats'].max()//dist['beats'].min()}:1")
ax[1].invert_yaxis()
plt.tight_layout()
save_fig(fig, "01_class_distribution")

sig = load_signal("100")[0]
a   = wfdb.rdann("100", "atr", pn_dir="mitdb")
seg, t = sig[:10*FS], np.arange(10*FS)/FS
m = a.sample < 10*FS
fig, ax = plt.subplots(figsize=(13, 3))
ax.plot(t, seg, lw=0.9, color="#2F3C7E")
for p, s in zip(a.sample[m], np.array(a.symbol)[m]):
    ax.axvline(p/FS, color="#990011", alpha=0.3, lw=0.8)
    ax.text(p/FS, seg.max()*1.08, s, ha="center", fontsize=9, color="#990011")
ax.set_xlabel("seconds"); ax.set_ylabel("mV")
ax.set_title("record 100, lead MLII, first 10 s with beat annotations")
plt.tight_layout()
save_fig(fig, "02_signal_annotated")

PRE, POST = int(0.25*FS), int(0.45*FS)
WIN = PRE + POST

collected = {c: [] for c in CLASSES}
for rec in ["100","106","200","208","213","232"]:
    s_, a_ = load_signal(rec)[0], wfdb.rdann(rec, "atr", pn_dir="mitdb")
    for samp, symv in zip(a_.sample, a_.symbol):
        if symv not in BEAT_SYMS:
            continue
        c = SYM2AAMI[symv]
        if c not in CLASSES or len(collected[c]) >= 400:
            continue
        if samp-PRE < 0 or samp+POST >= len(s_):
            continue
        b = s_[samp-PRE:samp+POST]
        collected[c].append((b - b.mean())/(b.std()+1e-9))

fig, axes = plt.subplots(1, 4, figsize=(15, 3.2), sharey=True)
for ax, c in zip(axes, CLASSES):
    arr = np.array(collected[c])
    tt = (np.arange(arr.shape[1]) - PRE)/FS*1000
    ax.plot(tt, arr.T, color="gray", alpha=0.03, lw=0.5)
    ax.plot(tt, arr.mean(0), color=CLASS_COL[c], lw=2.2)
    ax.set_title(f"{c} - {CLASS_NAME[c]}  (n={len(arr)})", fontsize=11)
    ax.set_xlabel("ms from R peak"); ax.grid(alpha=0.2)
axes[0].set_ylabel("normalised amplitude")
plt.tight_layout()
save_fig(fig, "03_beat_morphology")


# ============================================================================
# 4. Data cleaning
# ============================================================================

B, A = butter(3, [0.5/(FS/2), 40/(FS/2)], btype="band")

def clean(x):
    base = medfilt(medfilt(x, int(0.2*FS)|1), int(0.6*FS)|1)
    return filtfilt(B, A, x - base).astype(np.float32)

steps = pd.DataFrame([
    {"step": "Baseline removal", "method": "median filter 200 ms then 600 ms",
     "removes": "respiration drift"},
    {"step": "Bandpass",         "method": "Butterworth order 3, 0.5-40 Hz, zero phase",
     "removes": "power-line hum, muscle noise"},
    {"step": "Segmentation",     "method": f"{PRE} samples before R, {POST} after ({WIN} total, {WIN/FS*1000:.0f} ms)",
     "removes": "inter-beat gaps"},
    {"step": "Normalisation",    "method": "per beat, zero mean unit variance",
     "removes": "amplitude differences between recordings"},
])
save_table(steps, "04_preprocessing_steps")

raw = load_signal("108")[0]
cl  = clean(raw)
fig, ax = plt.subplots(2, 1, figsize=(13, 4.6), sharex=True)
tt = np.arange(10*FS)/FS
ax[0].plot(tt, raw[:10*FS], lw=0.9, color="#888780")
ax[0].set_title("record 108 raw"); ax[0].set_ylabel("mV")
ax[1].plot(tt, cl[:10*FS], lw=0.9, color="#1D9E75")
ax[1].set_title("baseline removed and bandpassed 0.5-40 Hz")
ax[1].set_xlabel("seconds"); ax[1].set_ylabel("mV")
plt.tight_layout()
save_fig(fig, "04_cleaning")


# ============================================================================
# 5. Feature engineering
# ============================================================================

def segment(rec):
    x = clean(load_signal(rec)[0])
    a = wfdb.rdann(rec, "atr", pn_dir="mitdb")
    samp = np.array([s for s, y in zip(a.sample, a.symbol) if y in BEAT_SYMS])
    syms = np.array([y for y in a.symbol if y in BEAT_SYMS])
    rr   = np.diff(samp)/FS
    grr  = float(np.median(rr)) if len(rr) else 1.0
    out = []
    for i, (s, y) in enumerate(zip(samp, syms)):
        c = SYM2AAMI[y]
        if c not in CLASSES or s-PRE < 0 or s+POST >= len(x):
            continue
        b = x[s-PRE:s+POST]
        b = (b - b.mean())/(b.std()+1e-9)
        pre  = rr[i-1] if i >= 1 else grr
        post = rr[i]   if i < len(rr) else grr
        lo   = max(0, i-10)
        loc  = float(np.mean(rr[lo:i])) if i > lo else grr
        out.append((b, (pre, post, loc, grr), c, rec, i, y))
    return out

def extract(b, rr):
    f = {}
    pre, post, loc, glob = rr
    f["pre_rr"], f["post_rr"], f["local_rr"], f["global_rr"] = pre, post, loc, glob
    f["pre_over_local"]    = pre /(loc +1e-9)
    f["pre_over_global"]   = pre /(glob+1e-9)
    f["post_over_pre"]     = post/(pre +1e-9)
    f["local_over_global"] = loc /(glob+1e-9)
    ds = np.interp(np.linspace(0, len(b)-1, 32), np.arange(len(b)), b)
    for i, v in enumerate(ds):
        f["morph%d" % i] = float(v)
    for i, c in enumerate(pywt.wavedec(b, "db4", level=4)):
        f["w%d_energy" % i] = float(np.sum(c**2))
        f["w%d_std" % i]    = float(np.std(c))
        f["w%d_mean" % i]   = float(np.mean(c))
    f["skew"], f["kurt"] = float(skew(b)), float(kurtosis(b))
    f["energy"], f["rpeak_amp"] = float(np.sum(b**2)), float(b.max())
    f["min_amp"] = float(b.min())
    f["qrs_width"] = float(np.sum(np.abs(b) > 0.5)/FS)
    return f

catalogue = pd.DataFrame([
    {"family": "Timing (RR)", "n": 8,
     "features": "pre_rr, post_rr, local_rr, global_rr + 4 ratios",
     "purpose": "is this beat early or late relative to the patient's own rhythm"},
    {"family": "Beat shape", "n": 32,
     "features": "morph0 .. morph31",
     "purpose": "the 700 ms curve resampled to 32 points"},
    {"family": "Wavelet", "n": 15,
     "features": "w0..w4 energy, std, mean (db4, level 4)",
     "purpose": "energy at five frequency scales; wide beats shift energy to coarse bands"},
    {"family": "Statistics", "n": 6,
     "features": "skew, kurt, energy, rpeak_amp, min_amp, qrs_width",
     "purpose": "sharpness, asymmetry and a direct width measurement"},
    {"family": "TOTAL", "n": 61, "features": "", "purpose": ""},
])
save_table(catalogue, "05_feature_catalogue")

if os.path.exists(FEAT) and not REBUILD_FEATURES:
    feat = pd.read_parquet(FEAT)
    print("loaded cached features:", feat.shape)
else:
    rows, t0 = [], time.time()
    for rec in tqdm(RECORDS, desc="features"):
        for b, rr, c, r, idx, symv in segment(rec):
            try:
                f = extract(b, rr)
            except Exception:
                continue
            f.update(record=r, beat_idx=idx, symbol=symv, aami=c,
                     split="DS1" if r in DS1 else "DS2")
            rows.append(f)
    feat = pd.DataFrame(rows)
    feat.to_parquet(FEAT, index=False)
    print("extracted %d beats in %.0f s" % (len(feat), time.time()-t0))

META     = ["record","beat_idx","symbol","aami","split"]
FEATURES = [c for c in feat.columns if c not in META]
print("beats:", len(feat), "| features designed:", len(FEATURES))


# ============================================================================
# 6. Data validation
# ============================================================================

feat[FEATURES] = feat[FEATURES].replace([np.inf, -np.inf], np.nan)
nan_before = int(feat[FEATURES].isna().sum().sum())
feat[FEATURES] = feat[FEATURES].fillna(feat[FEATURES].median())

overlap = len(set(feat[feat.split=="DS1"].record) & set(feat[feat.split=="DS2"].record))
checks = pd.DataFrame([
    {"check": "Beats extracted",            "value": f"{len(feat):,}", "status": "ok"},
    {"check": "Records",                    "value": feat["record"].nunique(), "status": "ok"},
    {"check": "Features designed",          "value": len(FEATURES), "status": "ok"},
    {"check": "Missing or infinite values", "value": nan_before,
     "status": "ok" if nan_before == 0 else "imputed with median"},
    {"check": "Constant features",          "value": int((feat[FEATURES].nunique() <= 1).sum()), "status": "ok"},
    {"check": "Classes present",            "value": ", ".join(sorted(feat['aami'].unique())), "status": "ok"},
    {"check": "Patients in both DS1 and DS2","value": overlap,
     "status": "ok" if overlap == 0 else "LEAK"},
])
save_table(checks, "06_data_validation")
assert overlap == 0

conc = []
for c in ["S", "F"]:
    for sp in ["DS1", "DS2"]:
        s = feat[(feat.aami==c) & (feat.split==sp)]["record"].value_counts()
        if len(s) == 0:
            continue
        conc.append({"class": c, "split": sp, "beats": int(s.sum()), "records": len(s),
                     "top_record": s.index[0], "top_beats": int(s.iloc[0]),
                     "top_share_pct": round(100*s.iloc[0]/s.sum(), 1),
                     "top3_share_pct": round(100*s.head(3).sum()/s.sum(), 1)})
conc = pd.DataFrame(conc)
save_table(conc, "07_minority_concentration")

fig, ax = plt.subplots(figsize=(8, 3.4))
lbl = conc["class"] + " in " + conc["split"]
ax.barh(lbl, conc["top_share_pct"], color="#990011", label="single worst record")
ax.barh(lbl, conc["top3_share_pct"] - conc["top_share_pct"],
        left=conc["top_share_pct"], color="#D85A30", alpha=0.6, label="next two records")
for i, r in conc.iterrows():
    ax.text(r["top_share_pct"]/2, i, f"rec {r['top_record']}", va="center", ha="center",
            color="white", fontsize=9)
ax.set_xlabel("share of the class held by the top records (%)")
ax.set_xlim(0, 100); ax.legend(loc="lower right", fontsize=9)
ax.set_title("minority classes are concentrated in very few patients")
ax.invert_yaxis(); plt.tight_layout()
save_fig(fig, "05_minority_concentration")


# ============================================================================
# 7. Feature selection
# ============================================================================

corr    = feat[FEATURES].corr().abs()
upper   = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
dropped = [c for c in upper.columns if (upper[c] > 0.97).any()]
FEATURES = [c for c in FEATURES if c not in dropped]

FAMILIES = {
    "rr":      [c for c in FEATURES if c in RR_NAMES],
    "morph":   [c for c in FEATURES if c.startswith("morph")],
    "wavelet": [c for c in FEATURES if c[0] == "w" and "_" in c],
    "stats":   [c for c in FEATURES if c in STAT_NAMES],
}
FAMILIES["ALL"] = FEATURES
assigned = set().union(*[set(v) for k, v in FAMILIES.items() if k != "ALL"])
missing  = [c for c in FEATURES if c not in assigned]
assert not missing, missing

sel = pd.DataFrame([{"family": k, "kept": len(v)} for k, v in FAMILIES.items() if k != "ALL"])
sel.loc[len(sel)] = ["TOTAL kept", len(FEATURES)]
sel.loc[len(sel)] = ["dropped (corr > 0.97)", len(dropped)]
save_table(sel, "08_feature_selection")
print()
print("dropped:", dropped)

X_all = feat[FEATURES].values
y_all = feat["aami"].values
g_all = feat["record"].values

ORD = FAMILIES["rr"] + FAMILIES["stats"] + FAMILIES["wavelet"] + FAMILIES["morph"]
C = feat[ORD].corr().abs().values
bounds, acc = [], 0
for k in ["rr", "stats", "wavelet", "morph"]:
    acc += len(FAMILIES[k]); bounds.append(acc)

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
im = ax[0].imshow(C, cmap="RdYlBu_r", vmin=0, vmax=1)
for b in bounds[:-1]:
    ax[0].axhline(b-0.5, color="black", lw=1.2)
    ax[0].axvline(b-0.5, color="black", lw=1.2)
mids = [0] + bounds
names   = ["rr", "stats", "wavelet", "morph"]
centres = [(mids[i] + mids[i+1])/2 - 0.5 for i in range(len(names))]
ax[0].set_xticks(centres); ax[0].set_xticklabels(names, fontsize=9)
ax[0].set_yticks(centres); ax[0].set_yticklabels(names, fontsize=9, rotation=90, va="center")
ax[0].tick_params(length=0); ax[0].grid(False)
ax[0].set_title("absolute correlation between the %d kept features" % len(FEATURES), pad=12)
fig.colorbar(im, ax=ax[0], fraction=0.046)

fstat = {}
for c in FEATURES:
    groups = [feat[feat.aami == k][c].values for k in CLASSES]
    gm = feat[c].mean()
    between = sum(len(g)*(g.mean()-gm)**2 for g in groups)/(len(CLASSES)-1)
    within  = sum(((g-g.mean())**2).sum() for g in groups)/(len(feat)-len(CLASSES))
    fstat[c] = between/(within+1e-12)
fs = (pd.Series(fstat).sort_values(ascending=False).rename("f_statistic")
        .reset_index().rename(columns={"index": "feature"}))
fs["family"] = fs["feature"].map({c: k for k, v in FAMILIES.items() if k != "ALL" for c in v})
fs["f_statistic"] = fs["f_statistic"].round(2)
save_table(fs.head(15), "09_univariate_separation", show=False)

FAM_C = {"rr": "#2F3C7E", "morph": "#C87A2A", "wavelet": "#990011", "stats": "#5F7A52"}
tp = fs.head(15).iloc[::-1]
ax[1].barh(tp["feature"], tp["f_statistic"], color=[FAM_C[f] for f in tp["family"]])
ax[1].set_xlabel("one-way F statistic between the four classes")
ax[1].set_title("which single features separate the classes at all")
ax[1].tick_params(axis="y", labelsize=8)
hs = [plt.Rectangle((0,0),1,1,color=FAM_C[f]) for f in FAM_C]
ax[1].legend(hs, list(FAM_C), fontsize=8, loc="lower right")
plt.tight_layout()
save_fig(fig, "06_feature_correlation")

print(fs.head(10).to_string(index=False))


# ============================================================================
# 8. Validation strategy
# ============================================================================

def split_random(test_size=0.25):
    return train_test_split(np.arange(len(feat)), test_size=test_size,
                            stratify=y_all, random_state=SEED)

def split_interpatient():
    return (np.where(feat["split"].values == "DS1")[0],
            np.where(feat["split"].values == "DS2")[0])

def subsample(idx, cap=CAP):
    if len(idx) <= cap:
        return idx
    keep, _ = train_test_split(np.arange(len(idx)), train_size=cap,
                               stratify=y_all[idx], random_state=SEED)
    return idx[keep]

prot = []
for name, fn in [("random", split_random), ("inter-patient", split_interpatient)]:
    tr, te = fn()
    prot.append({"protocol": name, "train_beats": len(tr), "test_beats": len(te),
                 "train_patients": len(set(g_all[tr])), "test_patients": len(set(g_all[te])),
                 "patients_in_both": len(set(g_all[tr]) & set(g_all[te]))})
prot = pd.DataFrame(prot)
save_table(prot, "10_validation_protocols")

tr_i, te_i = split_interpatient()
assert not (set(g_all[tr_i]) & set(g_all[te_i]))


# ============================================================================
# 9. Model training
# ============================================================================

MODEL_DESC = pd.DataFrame([
    {"model": "logreg", "name": "Logistic regression", "boundary": "one straight boundary",
     "role": "baseline every other model must beat"},
    {"model": "linsvm", "name": "Linear SVM", "boundary": "straight, widest margin",
     "role": "tests whether better placement helps"},
    {"model": "rbfsvm", "name": "RBF SVM", "boundary": "curved",
     "role": "tests whether curvature helps"},
    {"model": "rf", "name": "Random forest", "boundary": "staircase of thresholds",
     "role": "ensemble, and reports feature importance"},
])
save_table(MODEL_DESC, "11_models")

def make_models(n):
    m = {
      "logreg": (Pipeline([("sc", StandardScaler()),
                 ("clf", LogisticRegression(max_iter=3000, class_weight="balanced",
                                            random_state=SEED))]),
                 {"clf__C": [0.01, 0.1, 1, 10]}),
      "linsvm": (Pipeline([("sc", StandardScaler()),
                 ("clf", LinearSVC(class_weight="balanced", max_iter=5000,
                                   random_state=SEED))]),
                 {"clf__C": [0.01, 0.1, 1]}),
      "rf":     (Pipeline([("clf", RandomForestClassifier(n_estimators=300,
                           class_weight="balanced_subsample", n_jobs=-1,
                           random_state=SEED))]),
                 {"clf__max_depth": [6, 8, 12, None], "clf__min_samples_leaf": [1, 5]}),
    }
    if n <= CAP:
        m["rbfsvm"] = (Pipeline([("sc", StandardScaler()),
                       ("clf", SVC(class_weight="balanced", random_state=SEED))]),
                       {"clf__C": [1, 10], "clf__gamma": ["scale"]})
    return m

GRID_LOG, RES_MODELS = {}, {}

def run_split(key, title, tr, te):
    tr = subsample(tr)
    Xtr, ytr, gtr = X_all[tr], y_all[tr], g_all[tr]
    Xte, yte = X_all[te], y_all[te]
    print(f"\n{title}   train={len(ytr)}  test={len(yte)}")
    out = {}
    for name, (pipe, grid) in make_models(len(ytr)).items():
        t0 = time.time()
        cv = StratifiedGroupKFold(n_splits=min(3, len(set(gtr))))
        gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1,
                          return_train_score=True)
        gs.fit(Xtr, ytr, groups=gtr)
        GRID_LOG[(key, name)] = pd.DataFrame(gs.cv_results_)
        best = gs.best_estimator_
        pred = best.predict(Xte)
        out[name] = {"model": best, "pred": pred, "params": gs.best_params_,
                     "cv_f1": gs.best_score_, "train_pred": best.predict(Xtr),
                     "ytr": ytr, "secs": time.time()-t0,
                     "proba": best.predict_proba(Xte) if hasattr(best, "predict_proba") else None}
        print(f"  {name:<8} cv_f1={gs.best_score_:.4f}  {gs.best_params_}  ({time.time()-t0:.0f}s)")
    return {"results": out, "ytest": yte, "test_idx": te, "train_idx": tr}

tr, te = split_random()
RES_MODELS["random"] = run_split(
    "random", "RANDOM SPLIT - the same patient appears on both sides", tr, te)
tr, te = split_interpatient()
RES_MODELS["inter-patient"] = run_split(
    "inter-patient", "INTER-PATIENT SPLIT - DS1 trains, DS2 tests, no patient shared", tr, te)

grid_rows = []
for (split, name), df in GRID_LOG.items():
    pcols = [c for c in df.columns if c.startswith("param_")]
    for point, (_, r) in enumerate(df.iterrows()):
        row = {"split": split, "model": name, "point": point,
               "params": ", ".join(f"{c.replace('param_clf__','')}={r[c]}" for c in pcols),
               "cv_train_f1": round(r["mean_train_score"], 4),
               "cv_test_f1":  round(r["mean_test_score"], 4),
               "cv_std":      round(r["std_test_score"], 4),
               "rank":        int(r["rank_test_score"])}
        grid_rows.append(row)
grid_df = pd.DataFrame(grid_rows).sort_values(["split","model","rank"])
save_table(grid_df, "12_hyperparameter_search", show=False)
print(grid_df[grid_df.split == "inter-patient"].to_string(index=False))


# ============================================================================
# 10. Model evaluation
# ============================================================================

def brier_multi(y_true, proba, classes):
    Y = np.zeros_like(proba)
    for i, c in enumerate(classes):
        Y[:, i] = (y_true == c).astype(float)
    return float(np.mean(np.sum((proba - Y)**2, axis=1)))

rows = []
for split, pack in RES_MODELS.items():
    yte = pack["ytest"]
    for name, r in pack["results"].items():
        pred = r["pred"]
        row = {"split": split, "model": name,
               "accuracy":    round(accuracy_score(yte, pred), 4),
               "bal_acc":     round(balanced_accuracy_score(yte, pred), 4),
               "f1_macro":    round(f1_score(yte, pred, average="macro", labels=CLASSES, zero_division=0), 4),
               "f1_weighted": round(f1_score(yte, pred, average="weighted", labels=CLASSES, zero_division=0), 4),
               "kappa":       round(cohen_kappa_score(yte, pred), 4),
               "train_f1":    round(f1_score(r["ytr"], r["train_pred"], average="macro", labels=CLASSES, zero_division=0), 4),
               "train_secs":  round(r["secs"], 1)}
        if r["proba"] is not None:
            row["log_loss"] = round(log_loss(yte, r["proba"], labels=list(r["model"].classes_)), 4)
            row["brier"]    = round(brier_multi(yte, r["proba"], list(r["model"].classes_)), 4)
        else:
            row["log_loss"], row["brier"] = np.nan, np.nan
        for c in CLASSES:
            row["f1_" + c] = round(f1_score(yte, pred, average=None, labels=CLASSES, zero_division=0)[CLASSES.index(c)], 3)
        rows.append(row)

metrics = pd.DataFrame(rows)
metrics["overfit_gap"] = (metrics["train_f1"] - metrics["f1_macro"]).round(4)
save_table(metrics, "13_model_metrics")

order = ["logreg", "linsvm", "rbfsvm", "rf"]
piv_acc = metrics.pivot(index="model", columns="split", values="accuracy").reindex(order)
piv_f1  = metrics.pivot(index="model", columns="split", values="f1_macro").reindex(order)
piv_gap = metrics.pivot(index="model", columns="split", values="overfit_gap").reindex(order)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, piv, title in zip(ax, [piv_acc, piv_f1, piv_gap],
                         ["accuracy", "macro F1", "overfitting gap"]):
    piv.plot.bar(ax=a, color=[SPLIT_COL[c] for c in piv.columns], width=0.75)
    a.set_title(title); a.set_xlabel(""); a.tick_params(axis="x", rotation=0)
    a.legend(fontsize=8)
    for cont in a.containers:
        a.bar_label(cont, fmt="%.3f", fontsize=7, padding=1)
ax[0].set_ylim(0, 1.12); ax[1].set_ylim(0, 1.12)
ax[2].set_ylim(0, max(piv_gap.max().max()*1.3, 0.1))
plt.tight_layout()
save_fig(fig, "07_model_comparison")

pc_rows = []
for split, pack in RES_MODELS.items():
    yte = pack["ytest"]
    name = max(pack["results"], key=lambda k: f1_score(yte, pack["results"][k]["pred"],
                                                       average="macro", labels=CLASSES, zero_division=0))
    p, rc, f1, sup = precision_recall_fscore_support(yte, pack["results"][name]["pred"],
                                                     labels=CLASSES, zero_division=0)
    for i, c in enumerate(CLASSES):
        pc_rows.append({"split": split, "best_model": name, "class": c,
                        "meaning": CLASS_NAME[c], "precision": round(p[i], 3),
                        "recall": round(rc[i], 3), "f1": round(f1[i], 3), "support": int(sup[i])})
per_class = pd.DataFrame(pc_rows)
save_table(per_class, "14_per_class_metrics")

piv_pc = per_class.pivot(index="class", columns="split", values="f1").reindex(CLASSES)
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
piv_pc.plot.bar(ax=ax[0], color=[SPLIT_COL[c] for c in piv_pc.columns], width=0.75)
ax[0].set_title("per-class F1 by protocol"); ax[0].set_ylim(0, 1.12)
ax[0].tick_params(axis="x", rotation=0); ax[0].set_xlabel("")
for cont in ax[0].containers:
    ax[0].bar_label(cont, fmt="%.3f", fontsize=8, padding=1)

delta = (piv_pc["inter-patient"] - piv_pc["random"])
ax[1].bar(delta.index, delta.values, color=[CLASS_COL[c] for c in delta.index], width=0.6)
ax[1].axhline(0, color="black", lw=0.8)
ax[1].set_title("change when patients are separated")
ax[1].set_ylabel("F1 change")
for i, v in enumerate(delta.values):
    ax[1].text(i, v - 0.05 if v < 0 else v + 0.02, f"{v:+.3f}", ha="center", fontsize=9)
ax[1].set_ylim(delta.min()*1.3, 0.12)
plt.tight_layout()
save_fig(fig, "08_per_class_f1")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (split, pack) in zip(axes, RES_MODELS.items()):
    yte = pack["ytest"]
    name = max(pack["results"], key=lambda k: f1_score(yte, pack["results"][k]["pred"],
                                                       average="macro", labels=CLASSES, zero_division=0))
    cm  = confusion_matrix(yte, pack["results"][name]["pred"], labels=CLASSES)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for a in range(len(CLASSES)):
        for b in range(len(CLASSES)):
            ax.text(b, a, f"{cm[a,b]:,}\n{cmn[a,b]*100:.1f}%", ha="center", va="center",
                    fontsize=9, color="white" if cmn[a,b] > 0.5 else "black")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES)
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{split} ({name})"); ax.grid(False)
plt.tight_layout()
save_fig(fig, "09_confusion_matrices")

cm_rows = []
for split, pack in RES_MODELS.items():
    yte = pack["ytest"]
    name = max(pack["results"], key=lambda k: f1_score(yte, pack["results"][k]["pred"],
                                                       average="macro", labels=CLASSES, zero_division=0))
    cm = confusion_matrix(yte, pack["results"][name]["pred"], labels=CLASSES)
    for i, tc in enumerate(CLASSES):
        for j, pc in enumerate(CLASSES):
            cm_rows.append({"split": split, "true": tc, "predicted": pc, "beats": int(cm[i, j])})
save_table(pd.DataFrame(cm_rows), "15_confusion_matrix", show=False)
print("confusion matrices saved")

pc_all = []
for split, pack in RES_MODELS.items():
    yte = pack["ytest"]
    for name, r in pack["results"].items():
        p, rc, f1v, sup = precision_recall_fscore_support(yte, r["pred"], labels=CLASSES,
                                                          zero_division=0)
        for i, c in enumerate(CLASSES):
            pc_all.append({"split": split, "model": name, "class": c,
                           "meaning": CLASS_NAME[c], "precision": round(p[i], 3),
                           "recall": round(rc[i], 3), "f1": round(f1v[i], 3),
                           "support": int(sup[i])})
pc_all = pd.DataFrame(pc_all)
save_table(pc_all, "16_per_class_all_models", show=False)
print(pc_all[pc_all.split == "inter-patient"].to_string(index=False))

MODEL_COL = {"logreg": "#2F3C7E", "linsvm": "#C87A2A",
             "rbfsvm": "#990011", "rf": "#5F7A52"}

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for row, split in enumerate(["random", "inter-patient"]):
    sub = pc_all[pc_all.split == split]
    for col, metric in enumerate(["precision", "recall", "f1"]):
        ax = axes[row, col]
        xs = np.arange(len(CLASSES))
        for i, m in enumerate(order):
            v = [sub[(sub.model == m) & (sub["class"] == c)][metric].iloc[0] for c in CLASSES]
            ax.bar(xs + (i - 1.5)*0.2, v, width=0.2, color=MODEL_COL[m], label=m)
        ax.set_xticks(xs); ax.set_xticklabels(CLASSES)
        ax.set_ylim(0, 1.08); ax.set_title(metric + " - " + split, fontsize=11)
        if row == 0 and col == 0:
            ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
save_fig(fig, "10_per_class_all_models")

fig, axes = plt.subplots(2, 4, figsize=(17, 8))
for row, (split, pack) in enumerate(RES_MODELS.items()):
    yte = pack["ytest"]
    for col, name in enumerate(order):
        ax   = axes[row, col]
        pred = pack["results"][name]["pred"]
        cm   = confusion_matrix(yte, pred, labels=CLASSES)
        cmn  = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for a in range(len(CLASSES)):
            for b in range(len(CLASSES)):
                ax.text(b, a, "%.0f%%" % (cmn[a, b]*100), ha="center", va="center",
                        fontsize=9, color="white" if cmn[a, b] > 0.5 else "black")
        ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, fontsize=9)
        ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES, fontsize=9)
        ax.grid(False)
        mf = f1_score(yte, pred, average="macro", labels=CLASSES, zero_division=0)
        ax.set_title("%s  macro F1 %.3f" % (name, mf), fontsize=10)
        if col == 0:
            ax.set_ylabel(split + chr(10) + "true", fontsize=10)
        if row == 1:
            ax.set_xlabel("predicted", fontsize=9)
plt.tight_layout()
save_fig(fig, "11_confusion_all_models")

gd = grid_df[grid_df.split == "inter-patient"].sort_values(["model", "point"])

fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
for ax, name in zip(axes, order):
    sub = gd[gd.model == name]
    xs  = range(len(sub))
    ax.plot(xs, sub["cv_train_f1"], "o-", color="#C87A2A", label="train")
    ax.plot(xs, sub["cv_test_f1"],  "o-", color="#990011", label="validation")
    ax.fill_between(xs, sub["cv_test_f1"], sub["cv_train_f1"], color="#C87A2A", alpha=0.12)
    ax.set_xticks(list(xs))
    lbl = (sub["params"].str.replace("min_samples_leaf", "leaf", regex=False)
                        .str.replace("max_depth", "depth", regex=False)
                        .str.replace(", gamma=scale", "", regex=False))
    ax.set_xticklabels(lbl, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_title(name, fontsize=11)
    ax.set_ylabel("macro F1" if name == order[0] else "")
    ax.legend(fontsize=8)
fig.suptitle("train vs validation across each model's own hyperparameter sweep, "
             "grouped 3-fold CV on the training patients", fontsize=11)
plt.tight_layout()
save_fig(fig, "12_hyperparameter_curves")


# ============================================================================
# 11. Feature importance
# ============================================================================

FAM_OF = {c: k for k, v in FAMILIES.items() if k != "ALL" for c in v}
FAM_COL = {"rr": "#2F3C7E", "morph": "#C87A2A", "wavelet": "#990011", "stats": "#5F7A52"}

rf_best = RES_MODELS["inter-patient"]["results"]["rf"]["model"].named_steps["clf"]
imp = pd.DataFrame({"feature": FEATURES,
                    "family": [FAM_OF[c] for c in FEATURES],
                    "importance": rf_best.feature_importances_.round(5)})
imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
imp["rank"] = imp.index + 1
save_table(imp.head(20)[["rank","feature","family","importance"]], "17_top_features")

fam_imp = (imp.groupby("family")["importance"].agg(["sum","count","mean"])
             .rename(columns={"sum":"total_importance","count":"n_features","mean":"mean_importance"})
             .sort_values("total_importance", ascending=False).round(5).reset_index())
fam_imp["importance_per_feature_rank"] = fam_imp["mean_importance"].rank(ascending=False).astype(int)
save_table(fam_imp, "18_family_importance")

top = imp.head(20).iloc[::-1]
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].barh(top["feature"], top["importance"], color=[FAM_COL[f] for f in top["family"]])
ax[0].set_xlabel("impurity importance"); ax[0].set_title("top 20 features")
ax[0].tick_params(axis="y", labelsize=8)
hs = [plt.Rectangle((0,0),1,1,color=FAM_COL[f]) for f in FAM_COL]
ax[0].legend(hs, list(FAM_COL), fontsize=8, loc="lower right")

ax[1].bar(fam_imp["family"], fam_imp["total_importance"],
          color=[FAM_COL[f] for f in fam_imp["family"]], width=0.6)
for i, r in fam_imp.iterrows():
    ax[1].text(i, r["total_importance"]+0.01,
               f"{r['total_importance']:.3f}\n{int(r['n_features'])} feat", ha="center", fontsize=9)
ax[1].set_ylabel("summed importance"); ax[1].set_title("importance by family")
ax[1].set_ylim(0, fam_imp["total_importance"].max()*1.3)
plt.tight_layout()
save_fig(fig, "13_feature_importance")


# ============================================================================
# 12. Feature family ablation
# ============================================================================

def quick_rf(cols, tr, te, trees=200, depth=None, leaf=1, sampler=None,
             weight="balanced_subsample", y_override=None):
    ci = [FEATURES.index(c) for c in cols]
    y_src = y_all if y_override is None else y_override
    Xtr, ytr = X_all[np.ix_(tr, ci)], y_src[tr]
    if sampler is not None:
        Xtr, ytr = sampler.fit_resample(Xtr, ytr)
    m = RandomForestClassifier(n_estimators=trees, max_depth=depth, min_samples_leaf=leaf,
                               class_weight=weight, n_jobs=-1, random_state=SEED)
    m.fit(Xtr, ytr)
    pred = m.predict(X_all[np.ix_(te, ci)])
    yte  = y_src[te]
    per  = f1_score(yte, pred, average=None, labels=CLASSES, zero_division=0)
    return {"f1_macro": round(f1_score(yte, pred, average="macro", labels=CLASSES, zero_division=0), 4),
            "accuracy": round(accuracy_score(yte, pred), 4),
            **{"f1_" + c: round(per[i], 4) for i, c in enumerate(CLASSES)},
            "n_train": len(ytr), "pred": pred}

tr_i, te_i = split_interpatient()
tr_s = subsample(tr_i)

abl = []
for fam in ["rr", "morph", "wavelet", "stats", "ALL"]:
    r = quick_rf(FAMILIES[fam], tr_s, te_i)
    r.pop("pred")
    r.pop("n_train")
    abl.append({"family": fam, "n_features": len(FAMILIES[fam]), **r})
    print(f"  {fam:<8} macro F1 {r['f1_macro']:.4f}")
abl = pd.DataFrame(abl)
save_table(abl, "19_family_ablation")

fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
ax[0].bar(abl["family"], abl["f1_macro"],
          color=[FAM_COL.get(f, "#888780") for f in abl["family"]], width=0.6)
for i, v in enumerate(abl["f1_macro"]):
    ax[0].text(i, v+0.008, f"{v:.3f}", ha="center", fontsize=9)
ax[0].set_ylabel("macro F1"); ax[0].set_title("each family trained alone")
ax[0].set_ylim(0, abl["f1_macro"].max()*1.25)

w, xs = 0.2, np.arange(len(abl))
for i, c in enumerate(CLASSES):
    ax[1].bar(xs + i*w - 1.5*w, abl["f1_" + c], width=w, color=CLASS_COL[c], label=c)
ax[1].set_xticks(xs); ax[1].set_xticklabels(abl["family"])
ax[1].set_ylabel("F1"); ax[1].set_title("per-class F1 by family")
ax[1].legend(fontsize=9)
plt.tight_layout()
save_fig(fig, "14_family_ablation")


# ============================================================================
# 13. Feature combination study
# ============================================================================

COMBOS = {
    "rr":                 FAMILIES["rr"],
    "morph":              FAMILIES["morph"],
    "rr+stats":           FAMILIES["rr"] + FAMILIES["stats"],
    "rr+wavelet":         FAMILIES["rr"] + FAMILIES["wavelet"],
    "rr+morph":           FAMILIES["rr"] + FAMILIES["morph"],
    "rr+wavelet+stats":   FAMILIES["rr"] + FAMILIES["wavelet"] + FAMILIES["stats"],
    "morph+wavelet":      FAMILIES["morph"] + FAMILIES["wavelet"],
    "ALL":                FAMILIES["ALL"],
}

comb = []
for name, cols in COMBOS.items():
    r = quick_rf(cols, tr_s, te_i); r.pop("pred")
    r.pop("n_train")
    comb.append({"combination": name, "n_features": len(cols), **r})
    print(f"  {name:<18} macro F1 {r['f1_macro']:.4f}   F1(S) {r['f1_S']:.4f}")
comb = pd.DataFrame(comb).sort_values("f1_macro", ascending=False).reset_index(drop=True)
save_table(comb, "20_feature_combinations")

best_c = comb.iloc[0]
all_c  = comb[comb.combination == "ALL"].iloc[0]
verdict = pd.DataFrame([
    {"question": "Best combination", "answer": best_c["combination"],
     "f1_macro": best_c["f1_macro"], "f1_S": best_c["f1_S"]},
    {"question": "Using every feature", "answer": "ALL",
     "f1_macro": all_c["f1_macro"], "f1_S": all_c["f1_S"]},
    {"question": "Cost of adding the rest",
     "answer": f"{all_c['f1_macro'] - best_c['f1_macro']:+.4f} macro F1",
     "f1_macro": np.nan, "f1_S": round(all_c["f1_S"] - best_c["f1_S"], 4)},
])
save_table(verdict, "21_combination_verdict")

fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
cb = comb.sort_values("f1_macro")
cols = ["#990011" if c == "ALL" else "#2F3C7E" for c in cb["combination"]]
ax[0].barh(cb["combination"], cb["f1_macro"], color=cols)
for i, v in enumerate(cb["f1_macro"]):
    ax[0].text(v+0.005, i, f"{v:.3f}", va="center", fontsize=9)
ax[0].set_xlabel("macro F1"); ax[0].set_title("feature combinations")
ax[0].set_xlim(0, cb["f1_macro"].max()*1.2)

ax[1].barh(cb["combination"], cb["f1_S"], color=cols)
for i, v in enumerate(cb["f1_S"]):
    ax[1].text(v+0.005, i, f"{v:.3f}", va="center", fontsize=9)
ax[1].set_xlabel("F1 on class S"); ax[1].set_title("the class that separates the combinations")
ax[1].set_xlim(0, max(cb["f1_S"].max()*1.25, 0.05))
plt.tight_layout()
save_fig(fig, "15_feature_combinations")


# ============================================================================
# 14. Is class imbalance the cause
# ============================================================================

IMB = {
    "none (class_weight off)":   (None, None),
    "class_weight balanced":     (None, "balanced_subsample"),
    "random oversample":         (RandomOverSampler(random_state=SEED), None),
    "SMOTE":                     (SMOTE(random_state=SEED, k_neighbors=5), None),
    "random undersample":        (RandomUnderSampler(random_state=SEED), None),
}

imb = []
for name, (samp, weight) in IMB.items():
    r = quick_rf(FEATURES, tr_s, te_i, sampler=samp, weight=weight)
    r.pop("pred")
    ratio = "1:1" if samp is not None else f"{(y_all[tr_s]=='N').sum()//max((y_all[tr_s]=='F').sum(),1)}:1"
    imb.append({"strategy": name, "train_rows": r.pop("n_train"),
                "N_to_F_ratio": ratio, **r})
    print(f"  {name:<26} macro F1 {r['f1_macro']:.4f}   F1(S) {r['f1_S']:.4f}")
imb = pd.DataFrame(imb)
save_table(imb, "22_imbalance_experiment")

spread = imb["f1_S"].max() - imb["f1_S"].min()
base   = imb[imb.strategy == "class_weight balanced"]["f1_S"].iloc[0]
iv = pd.DataFrame([
    {"finding": "Spread in F1(S) across all five strategies", "value": round(spread, 4)},
    {"finding": "Spread in macro F1 across all five strategies",
     "value": round(imb["f1_macro"].max() - imb["f1_macro"].min(), 4)},
    {"finding": "Best F1(S) achieved by any balancing",
     "value": f"{imb['f1_S'].max():.4f} ({imb.loc[imb.f1_S.idxmax(),'strategy']})"},
    {"finding": "F1(S) from feature choice instead (section 13)",
     "value": f"{comb['f1_S'].max():.4f} ({comb.loc[comb.f1_S.idxmax(),'combination']})"},
    {"finding": "Conclusion",
     "value": "imbalance is not the bottleneck" if spread < 0.05 else "imbalance matters"},
])
save_table(iv, "23_imbalance_verdict")

fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
ib = imb.iloc[::-1]
ax[0].barh(ib["strategy"], ib["f1_macro"], color="#2F3C7E")
for i, v in enumerate(ib["f1_macro"]):
    ax[0].text(v+0.004, i, f"{v:.3f}", va="center", fontsize=9)
ax[0].set_xlabel("macro F1"); ax[0].set_title("balancing strategies")
ax[0].set_xlim(0, ib["f1_macro"].max()*1.2)

w, xs = 0.2, np.arange(len(imb))
for i, c in enumerate(CLASSES):
    ax[1].bar(xs + i*w - 1.5*w, imb["f1_" + c], width=w, color=CLASS_COL[c], label=c)
ax[1].set_xticks(xs)
ax[1].set_xticklabels([s.replace(" ", "\n") for s in imb["strategy"]], fontsize=8)
ax[1].set_ylabel("F1"); ax[1].set_title("per-class F1 across balancing strategies")
ax[1].set_ylim(0, max(imb[["f1_" + c for c in CLASSES]].max().max()*1.35, 0.1))
ax[1].legend(fontsize=9, ncol=4, loc="upper center")
plt.tight_layout()
save_fig(fig, "16_imbalance_experiment")


# ============================================================================
# 15. Learning curve
# ============================================================================

FRACS = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
lc = []
for f in FRACS:
    n = max(int(len(tr_s)*f), 200)
    sub = tr_s if f == 1.0 else subsample(tr_s, n)
    r = quick_rf(FEATURES, sub, te_i); r.pop("pred")
    ntr_f = quick_rf(FEATURES, sub, sub)
    lc.append({"fraction": f, "train_beats": len(sub),
               "train_patients": len(set(g_all[sub])),
               "train_f1": ntr_f["f1_macro"], "test_f1": r["f1_macro"],
               "test_acc": r["accuracy"], "f1_S": r["f1_S"], "f1_V": r["f1_V"]})
    print(f"  {len(sub):>6} beats -> test macro F1 {r['f1_macro']:.4f}")
lc = pd.DataFrame(lc)
lc["gap"] = (lc["train_f1"] - lc["test_f1"]).round(4)
save_table(lc, "24_learning_curve")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
ax[0].plot(lc["train_beats"], lc["train_f1"], "o-", color="#C87A2A", label="train")
ax[0].plot(lc["train_beats"], lc["test_f1"], "o-", color="#990011", label="test (DS2)")
ax[0].fill_between(lc["train_beats"], lc["test_f1"], lc["train_f1"], color="#C87A2A", alpha=0.12)
ax[0].set_xscale("log"); ax[0].set_xlabel("training beats"); ax[0].set_ylabel("macro F1")
ax[0].set_title("learning curve"); ax[0].legend(fontsize=9); ax[0].set_ylim(0, 1.05)

for c, col in [("f1_S", CLASS_COL["S"]), ("f1_V", CLASS_COL["V"])]:
    ax[1].plot(lc["train_beats"], lc[c], "o-", color=col, label=c.replace("f1_", "class "))
ax[1].set_xscale("log"); ax[1].set_xlabel("training beats"); ax[1].set_ylabel("F1")
ax[1].set_title("more beats do not help the rare class"); ax[1].legend(fontsize=9)
plt.tight_layout()
save_fig(fig, "17_learning_curve")

gain = pd.DataFrame([{
    "data_increase": f"{lc['train_beats'].iloc[-1]/lc['train_beats'].iloc[0]:.0f}x",
    "macro_f1_gain": round(lc["test_f1"].iloc[-1] - lc["test_f1"].iloc[0], 4),
    "f1_S_gain":     round(lc["f1_S"].iloc[-1] - lc["f1_S"].iloc[0], 4),
    "patients_gain": int(lc["train_patients"].iloc[-1] - lc["train_patients"].iloc[0]),
}])
save_table(gain, "25_learning_curve_gain")


# ============================================================================
# 16. Validation curves
# ============================================================================

DEPTHS = [2, 4, 6, 8, 12, None]
vc = []
for d in DEPTHS:
    te_r = quick_rf(FEATURES, tr_s, te_i, depth=d)
    tr_r = quick_rf(FEATURES, tr_s, tr_s, depth=d)
    vc.append({"max_depth": "None" if d is None else d,
               "train_f1": tr_r["f1_macro"], "test_f1": te_r["f1_macro"],
               "gap": round(tr_r["f1_macro"] - te_r["f1_macro"], 4),
               "f1_S": te_r["f1_S"], "accuracy": te_r["accuracy"]})
    print(f"  depth {str(d):<5} train {tr_r['f1_macro']:.4f}  test {te_r['f1_macro']:.4f}")
vc = pd.DataFrame(vc)
save_table(vc, "26_depth_curve")

TREES = [10, 50, 100, 200, 400]
tc = []
for n in TREES:
    te_r = quick_rf(FEATURES, tr_s, te_i, trees=n)
    tr_r = quick_rf(FEATURES, tr_s, tr_s, trees=n)
    tc.append({"n_estimators": n, "train_f1": tr_r["f1_macro"], "test_f1": te_r["f1_macro"],
               "gap": round(tr_r["f1_macro"] - te_r["f1_macro"], 4)})
tc = pd.DataFrame(tc)
save_table(tc, "27_tree_count_curve")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
xs = range(len(vc))
ax[0].plot(xs, vc["train_f1"], "o-", color="#C87A2A", label="train")
ax[0].plot(xs, vc["test_f1"], "o-", color="#990011", label="test (DS2)")
ax[0].fill_between(xs, vc["test_f1"], vc["train_f1"], color="#C87A2A", alpha=0.12)
for i, g in enumerate(vc["gap"]):
    ax[0].text(i, max(vc["test_f1"].iloc[i] - 0.08, 0.02), f"gap {g:.2f}",
               ha="center", fontsize=8, color="#888780")
ax[0].set_xticks(list(xs)); ax[0].set_xticklabels(vc["max_depth"])
ax[0].set_xlabel("max_depth"); ax[0].set_ylabel("macro F1")
ax[0].set_title("depth drives the overfitting gap"); ax[0].legend(fontsize=9); ax[0].set_ylim(0, 1.05)

ax[1].plot(tc["n_estimators"], tc["train_f1"], "o-", color="#C87A2A", label="train")
ax[1].plot(tc["n_estimators"], tc["test_f1"], "o-", color="#990011", label="test (DS2)")
ax[1].set_xlabel("n_estimators"); ax[1].set_ylabel("macro F1")
ax[1].set_title("tree count does not"); ax[1].legend(fontsize=9); ax[1].set_ylim(0, 1.05)
plt.tight_layout()
save_fig(fig, "18_validation_curves")


# ============================================================================
# 17. Cross-validation
# ============================================================================

cvk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_rows = []
for k, (a, b) in enumerate(cvk.split(X_all, y_all, groups=g_all), 1):
    r = quick_rf(FEATURES, subsample(a), b); r.pop("pred")
    cv_rows.append({"fold": k, "train_patients": len(set(g_all[a])),
                    "test_patients": len(set(g_all[b])), "test_beats": len(b), **r})
    print(f"  fold {k}: macro F1 {r['f1_macro']:.4f}")
cv = pd.DataFrame(cv_rows).drop(columns=["n_train"])
summ = {"fold": "mean"}
for c in cv.columns[1:]:
    summ[c] = round(cv[c].mean(), 4)
sd = {"fold": "std"}
for c in cv.columns[1:]:
    sd[c] = round(cv[c].std(), 4)
cv = pd.concat([cv, pd.DataFrame([summ, sd])], ignore_index=True)
save_table(cv, "28_cross_validation")

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
folds = cv[cv.fold.apply(lambda x: isinstance(x, int))]
m, s = summ["f1_macro"], sd["f1_macro"]
ax[0].bar(folds["fold"], folds["f1_macro"], color="#2F3C7E", width=0.6)
ax[0].axhline(m, color="#990011", lw=1.6, label=f"mean {m:.3f}")
ax[0].fill_between([0.4, 5.6], m-s, m+s, color="#990011", alpha=0.12, label=f"+/- 1 sd ({s:.3f})")
for i, v in zip(folds["fold"], folds["f1_macro"]):
    ax[0].text(i, v+0.01, f"{v:.3f}", ha="center", fontsize=9)
ax[0].set_xlabel("fold"); ax[0].set_ylabel("macro F1"); ax[0].set_xlim(0.4, 5.6)
ax[0].set_ylim(0, max(folds["f1_macro"].max(), m + s)*1.18)
ax[0].set_title("5-fold grouped by patient"); ax[0].legend(fontsize=9)

for i, c in enumerate(CLASSES):
    ax[1].bar(folds["fold"] + (i-1.5)*0.2, folds["f1_" + c], width=0.2,
              color=CLASS_COL[c], label=c)
ax[1].set_xlabel("fold"); ax[1].set_ylabel("F1")
ax[1].set_title("per-class F1 by fold"); ax[1].set_ylim(0, 1.15)
ax[1].legend(fontsize=9, ncol=4, loc="upper center")
plt.tight_layout()
save_fig(fig, "19_cross_validation")


# ============================================================================
# 18. Per-record error analysis
# ============================================================================

best_name = max(RES_MODELS["inter-patient"]["results"],
                key=lambda k: f1_score(RES_MODELS["inter-patient"]["ytest"],
                                       RES_MODELS["inter-patient"]["results"][k]["pred"],
                                       average="macro", labels=CLASSES, zero_division=0))
pred_i = RES_MODELS["inter-patient"]["results"][best_name]["pred"]
dfe = pd.DataFrame({"record": g_all[te_i], "true": y_all[te_i], "pred": pred_i})

rec_rows = []
for rec, grp in dfe.groupby("record"):
    row = {"record": rec, "beats": len(grp),
           "accuracy": round((grp.true == grp.pred).mean(), 4)}
    for c in CLASSES:
        sub = grp[grp.true == c]
        row["n_" + c] = len(sub)
        row["recall_" + c] = round((sub.pred == c).mean(), 4) if len(sub) else np.nan
    rec_rows.append(row)
rec = pd.DataFrame(rec_rows).sort_values("accuracy")
save_table(rec, "29_per_record", show=False)
print(rec.head(10).to_string(index=False))

s_holders = (dfe[dfe.true == "S"].groupby("record").size()
               .sort_values(ascending=False).reset_index(name="S_beats"))
s_holders["share_of_DS2_S_pct"] = (100*s_holders["S_beats"]/s_holders["S_beats"].sum()).round(1)
s_holders = s_holders.merge(rec[["record","recall_S"]], on="record")
save_table(s_holders, "30_who_holds_class_S")

worst = s_holders.iloc[0]["record"]
keep_te = te_i[g_all[te_i] != worst]
r_with    = quick_rf(FEATURES, tr_s, te_i)
r_without = quick_rf(FEATURES, tr_s, keep_te)
excl = pd.DataFrame([
    {"test_set": "DS2 complete", "patients": len(set(g_all[te_i])), "beats": len(te_i),
     "f1_macro": r_with["f1_macro"], "f1_S": r_with["f1_S"], "f1_V": r_with["f1_V"]},
    {"test_set": f"DS2 without record {worst}", "patients": len(set(g_all[keep_te])),
     "beats": len(keep_te), "f1_macro": r_without["f1_macro"],
     "f1_S": r_without["f1_S"], "f1_V": r_without["f1_V"]},
    {"test_set": "change", "patients": -1, "beats": len(keep_te)-len(te_i),
     "f1_macro": round(r_without["f1_macro"]-r_with["f1_macro"], 4),
     "f1_S": round(r_without["f1_S"]-r_with["f1_S"], 4),
     "f1_V": round(r_without["f1_V"]-r_with["f1_V"], 4)},
])
save_table(excl, "31_single_patient_effect")

fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
sh = s_holders.head(8).iloc[::-1]
ax[0].barh(sh["record"].astype(str), sh["share_of_DS2_S_pct"], color=CLASS_COL["S"])
for i, r in enumerate(sh.itertuples()):
    ax[0].text(r.share_of_DS2_S_pct+1, i, f"{r.share_of_DS2_S_pct:.1f}%  recall {r.recall_S:.2f}",
               va="center", fontsize=8)
ax[0].set_xlabel("share of all class-S beats in DS2 (%)")
ax[0].set_title("class S lives in a handful of patients")
ax[0].set_xlim(0, 100)

rec_ok = rec.dropna(subset=["recall_S"]).sort_values("n_S", ascending=False)
ax[1].scatter(rec_ok["n_S"], rec_ok["recall_S"], s=40, color=CLASS_COL["S"], zorder=3)
mark = pd.concat([rec_ok.head(1), rec_ok.nsmallest(2, "recall_S")]).drop_duplicates("record")
for i, r in enumerate(mark.itertuples()):
    ax[1].annotate("record " + str(r.record), (r.n_S, r.recall_S), fontsize=9, ha="center",
                   fontweight="bold", textcoords="offset points",
                   xytext=(0, 12 if i % 2 == 0 else -18))
ax[1].set_xscale("symlog"); ax[1].set_xlabel("class-S beats in the record")
ax[1].set_ylabel("recall on class S"); ax[1].set_ylim(-0.15, 1.18)
ax[1].set_title("the patients with the most S beats are the ones it misses")
plt.tight_layout()
save_fig(fig, "20_per_record_analysis")


# ============================================================================
# 19. Honest model selection
# ============================================================================

inner = StratifiedGroupKFold(n_splits=3)
hsel_rows = []
for d in DEPTHS:
    scores = []
    for a, b in inner.split(X_all[tr_s], y_all[tr_s], groups=g_all[tr_s]):
        r = quick_rf(FEATURES, tr_s[a], tr_s[b], depth=d)
        scores.append(r["f1_macro"])
    hsel_rows.append({"max_depth": "None" if d is None else d,
                     "DS1_inner_cv_f1": round(float(np.mean(scores)), 4),
                     "DS1_inner_cv_sd": round(float(np.std(scores)), 4)})
    print(f"  depth {str(d):<5} DS1 inner CV {np.mean(scores):.4f}")
hsel = pd.DataFrame(hsel_rows)
hsel = hsel.merge(vc[["max_depth","test_f1"]].rename(columns={"test_f1":"DS2_f1"}),
                  on="max_depth", how="left")
chosen  = hsel.loc[hsel["DS1_inner_cv_f1"].idxmax()]
peeking = hsel.loc[hsel["DS2_f1"].idxmax()]
save_table(hsel, "32_honest_selection")

bias = pd.DataFrame([
    {"procedure": "Chosen on DS1 inner CV, reported on DS2",
     "max_depth": chosen["max_depth"], "reported_f1": chosen["DS2_f1"]},
    {"procedure": "Chosen by reading DS2 (what not to do)",
     "max_depth": peeking["max_depth"], "reported_f1": peeking["DS2_f1"]},
    {"procedure": "Optimism from letting the test set choose",
     "max_depth": "", "reported_f1": round(peeking["DS2_f1"] - chosen["DS2_f1"], 4)},
])
save_table(bias, "33_selection_bias")

fig, ax = plt.subplots(figsize=(8, 4))
xs = range(len(hsel))
ax.plot(xs, hsel["DS1_inner_cv_f1"], "o-", color="#2F3C7E", label="DS1 inner CV (used to choose)")
ax.plot(xs, hsel["DS2_f1"], "o-", color="#990011", label="DS2 (reported once)")
ax.axvline(list(hsel["max_depth"]).index(chosen["max_depth"]), color="#2F3C7E",
           ls="--", alpha=0.5, label=f"chosen: depth {chosen['max_depth']}")
ax.set_xticks(list(xs)); ax.set_xticklabels(hsel["max_depth"])
ax.set_xlabel("max_depth"); ax.set_ylabel("macro F1")
ax.set_title("selecting on DS1 and selecting on DS2 do not agree")
ax.legend(fontsize=9, loc="lower right"); ax.set_ylim(0, 1.08)
plt.tight_layout()
save_fig(fig, "21_honest_selection")


# ============================================================================
# 20. Three-class variant
# ============================================================================

CL3 = ["N", "S", "V"]
m3  = np.isin(y_all, CL3)
tr3 = np.array([i for i in tr_s if m3[i]])
te3 = np.array([i for i in te_i if m3[i]])

ci = list(range(len(FEATURES)))
m = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                           n_jobs=-1, random_state=SEED)
m.fit(X_all[np.ix_(tr3, ci)], y_all[tr3])
p3 = m.predict(X_all[np.ix_(te3, ci)])

pr, rc, f1v, sup = precision_recall_fscore_support(y_all[te3], p3, labels=CL3, zero_division=0)
three = pd.DataFrame({"class": CL3, "meaning": [CLASS_NAME[c] for c in CL3],
                      "precision": pr.round(3), "recall": rc.round(3),
                      "f1": f1v.round(3), "support": sup})
three.loc[len(three)] = ["MACRO", "", round(pr.mean(), 3), round(rc.mean(), 3),
                         round(f1v.mean(), 3), sup.sum()]
save_table(three, "34_three_class")

four_macro = metrics[(metrics.split=="inter-patient") & (metrics.model==best_name)]["f1_macro"].iloc[0]
cmp3 = pd.DataFrame([
    {"task": "4 classes (N/S/V/F)", "classes": 4, "macro_f1": four_macro},
    {"task": "3 classes (N/S/V)",   "classes": 3, "macro_f1": round(f1v.mean(), 4)},
    {"task": "difference", "classes": -1, "macro_f1": round(f1v.mean() - four_macro, 4)},
])
save_table(cmp3, "35_three_vs_four")

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
cm3  = confusion_matrix(y_all[te3], p3, labels=CL3)
cm3n = cm3 / cm3.sum(axis=1, keepdims=True)
ax[0].imshow(cm3n, cmap="Blues", vmin=0, vmax=1)
for a in range(3):
    for b in range(3):
        ax[0].text(b, a, f"{cm3[a,b]:,}\n{cm3n[a,b]*100:.1f}%", ha="center", va="center",
                   fontsize=9, color="white" if cm3n[a,b] > 0.5 else "black")
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(CL3)
ax[0].set_yticks(range(3)); ax[0].set_yticklabels(CL3)
ax[0].set_xlabel("predicted"); ax[0].set_ylabel("true")
ax[0].set_title("3-class confusion matrix"); ax[0].grid(False)

ax[1].bar(["4 classes", "3 classes"], [four_macro, f1v.mean()],
          color=["#888780", "#2F3C7E"], width=0.5)
for i, v in enumerate([four_macro, f1v.mean()]):
    ax[1].text(i, v+0.01, f"{v:.3f}", ha="center", fontsize=10)
ax[1].set_ylabel("macro F1"); ax[1].set_title("dropping class F")
ax[1].set_ylim(0, max(four_macro, f1v.mean())*1.3)
plt.tight_layout()
save_fig(fig, "22_three_class")


# ============================================================================
# 21. Pipeline validation
# ============================================================================

rng = np.random.default_rng(SEED)
y_shuf = y_all.copy()
y_shuf[tr_s] = rng.permutation(y_all[tr_s])
r_shuf = quick_rf(FEATURES, tr_s, te_i, y_override=y_shuf)

maj  = np.full(len(te_i), pd.Series(y_all[tr_s]).mode()[0])
rnd  = rng.choice(CLASSES, size=len(te_i))
prior = pd.Series(y_all[tr_s]).value_counts(normalize=True)
strat = rng.choice(prior.index.values, size=len(te_i), p=prior.values)
real  = quick_rf(FEATURES, tr_s, te_i)

def sc(pred):
    return (round(f1_score(y_all[te_i], pred, average="macro", labels=CLASSES, zero_division=0), 4),
            round(accuracy_score(y_all[te_i], pred), 4))

base = pd.DataFrame([
    {"baseline": "Always predict N",        "f1_macro": sc(maj)[0],   "accuracy": sc(maj)[1]},
    {"baseline": "Uniform random",          "f1_macro": sc(rnd)[0],   "accuracy": sc(rnd)[1]},
    {"baseline": "Random from class prior", "f1_macro": sc(strat)[0], "accuracy": sc(strat)[1]},
    {"baseline": "Labels shuffled in training", "f1_macro": r_shuf["f1_macro"],
     "accuracy": r_shuf["accuracy"]},
    {"baseline": "Real model",              "f1_macro": real["f1_macro"], "accuracy": real["accuracy"]},
])
base["beats_always_N_by"] = (base["f1_macro"] - sc(maj)[0]).round(4)
save_table(base, "36_pipeline_validation")

fig, ax = plt.subplots(figsize=(9, 3.8))
b = base.iloc[::-1]
cols = ["#990011" if s == "Real model" else "#888780" for s in b["baseline"]]
ax.barh(b["baseline"], b["f1_macro"], color=cols)
for i, v in enumerate(b["f1_macro"]):
    ax.text(v+0.004, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlabel("macro F1"); ax.set_title("the model against every trivial baseline")
ax.set_xlim(0, b["f1_macro"].max()*1.25)
plt.tight_layout()
save_fig(fig, "23_pipeline_validation")


# ============================================================================
# 22. Summary of findings
# ============================================================================

mi = metrics[metrics.split == "inter-patient"].sort_values("f1_macro", ascending=False).iloc[0]
mr = metrics[(metrics.split == "random") & (metrics.model == mi["model"])].iloc[0]
pc_i = per_class[per_class.split == "inter-patient"].set_index("class")
pc_r = per_class[per_class.split == "random"].set_index("class")
rr_f1S  = abl[abl.family == "rr"]["f1_S"].iloc[0]
oth_f1S = abl[abl.family.isin(["morph","wavelet","stats"])]["f1_S"].max()
best_cb = comb.iloc[0]
all_cb  = comb[comb.combination == "ALL"].iloc[0]
cv_mean = cv[cv.fold == "mean"]["f1_macro"].iloc[0]
cv_sd   = cv[cv.fold == "std"]["f1_macro"].iloc[0]
deep    = vc[vc.max_depth == "None"].iloc[0]
shallow = vc[vc.max_depth == 4].iloc[0]

findings = pd.DataFrame([
    {"n": 1, "finding": "Best model under the honest protocol",
     "evidence": f"{mi['model']}: accuracy {mi['accuracy']:.4f}, macro F1 {mi['f1_macro']:.4f}, kappa {mi['kappa']:.4f}",
     "table": "13_model_metrics", "figure": "07_model_comparison"},
    {"n": 2, "finding": "Accuracy hides the failure; macro F1 does not",
     "evidence": f"accuracy falls {(mr['accuracy']-mi['accuracy'])*100:+.1f} pts between protocols, macro F1 falls {(mr['f1_macro']-mi['f1_macro'])*100:+.1f} pts",
     "table": "13_model_metrics", "figure": "07_model_comparison"},
    {"n": 3, "finding": "Weighted F1 is accuracy wearing a different name",
     "evidence": f"weighted F1 {mi['f1_weighted']:.4f} vs accuracy {mi['accuracy']:.4f}, difference {abs(mi['f1_weighted']-mi['accuracy']):.4f}",
     "table": "13_model_metrics", "figure": "07_model_comparison"},
    {"n": 4, "finding": "Sharing patients between train and test inflates the rare classes",
     "evidence": f"F1(S) {pc_r.loc['S','f1']:.3f} -> {pc_i.loc['S','f1']:.3f}, F1(F) {pc_r.loc['F','f1']:.3f} -> {pc_i.loc['F','f1']:.3f}",
     "table": "14_per_class_metrics", "figure": "08_per_class_f1"},
    {"n": 5, "finding": "Class imbalance is not the bottleneck",
     "evidence": f"five balancing strategies move F1(S) by only {imb['f1_S'].max()-imb['f1_S'].min():.4f}",
     "table": "22_imbalance_experiment", "figure": "16_imbalance_experiment"},
    {"n": 6, "finding": "Only the timing features detect class S",
     "evidence": f"RR alone F1(S) {rr_f1S:.4f}, best of the other three families {oth_f1S:.4f}",
     "table": "19_family_ablation", "figure": "14_family_ablation"},
    {"n": 7, "finding": "Adding every feature is worse than choosing a subset",
     "evidence": f"{best_cb['combination']} macro F1 {best_cb['f1_macro']:.4f} vs ALL {all_cb['f1_macro']:.4f}",
     "table": "20_feature_combinations", "figure": "15_feature_combinations"},
    {"n": 8, "finding": "More beats do not help; more patients would",
     "evidence": f"{gain['data_increase'].iloc[0]} more training beats buys {gain['macro_f1_gain'].iloc[0]:+.4f} macro F1",
     "table": "24_learning_curve", "figure": "17_learning_curve"},
    {"n": 9, "finding": "Depth causes the overfitting, not tree count",
     "evidence": f"gap {shallow['gap']:.3f} at depth 4 vs {deep['gap']:.3f} unlimited; {tc['gap'].min():.3f} to {tc['gap'].max():.3f} across 10-400 trees",
     "table": "26_depth_curve", "figure": "18_validation_curves"},
    {"n": 10, "finding": "The result is one patient away from looking different",
     "evidence": f"excluding record {worst} changes F1(S) by {excl.iloc[2]['f1_S']:+.4f}",
     "table": "31_single_patient_effect", "figure": "20_per_record_analysis"},
    {"n": 11, "finding": "Choosing hyperparameters on the test set adds optimism",
     "evidence": f"{bias.iloc[2]['reported_f1']:+.4f} macro F1 from letting DS2 pick the depth",
     "table": "33_selection_bias", "figure": "21_honest_selection"},
    {"n": 12, "finding": "Fold-to-fold variation is larger than any tuning gain",
     "evidence": f"grouped 5-fold macro F1 {cv_mean:.4f} +/- {cv_sd:.4f}",
     "table": "28_cross_validation", "figure": "19_cross_validation"},
    {"n": 13, "finding": "The pipeline has no leakage of its own",
     "evidence": f"shuffled labels score {base[base.baseline.str.contains('shuffled')]['f1_macro'].iloc[0]:.4f} against the always-N baseline {base[base.baseline=='Always predict N']['f1_macro'].iloc[0]:.4f}",
     "table": "36_pipeline_validation", "figure": "23_pipeline_validation"},
    {"n": 14, "finding": "Dropping class F does not rescue the task",
     "evidence": f"3-class macro F1 {cmp3[cmp3.task=='3 classes (N/S/V)']['macro_f1'].iloc[0]:.4f} vs 4-class {four_macro:.4f}",
     "table": "35_three_vs_four", "figure": "22_three_class"},
])
save_table(findings, "37_summary_of_findings", show=False)
print(findings[["n","finding","evidence"]].to_string(index=False))

protocol_spread = abs(mr["f1_macro"] - mi["f1_macro"])
config_spread   = metrics[metrics.split == "inter-patient"]["f1_macro"].max() - \
                  metrics[metrics.split == "inter-patient"]["f1_macro"].min()
tuning_spread   = vc["test_f1"].max() - vc["test_f1"].min()

lever = pd.DataFrame([
    {"lever": "Evaluation protocol (random vs inter-patient)", "macro_f1_swing": round(protocol_spread, 4)},
    {"lever": "Choice of algorithm (4 models)",                "macro_f1_swing": round(config_spread, 4)},
    {"lever": "Feature subset (8 combinations)",               "macro_f1_swing": round(comb["f1_macro"].max()-comb["f1_macro"].min(), 4)},
    {"lever": "Tree depth (6 values)",                         "macro_f1_swing": round(tuning_spread, 4)},
    {"lever": "Class balancing (5 strategies)",                "macro_f1_swing": round(imb["f1_macro"].max()-imb["f1_macro"].min(), 4)},
    {"lever": "Tree count (10 to 400)",                        "macro_f1_swing": round(tc["test_f1"].max()-tc["test_f1"].min(), 4)},
]).sort_values("macro_f1_swing", ascending=False).reset_index(drop=True)
save_table(lever, "38_what_moves_the_result")

PROTO = "Evaluation protocol (random vs inter-patient)"
fig, ax = plt.subplots(figsize=(10, 4))
lv = lever.iloc[::-1]
cols = ["#990011" if l == PROTO else "#2F3C7E" for l in lv["lever"]]
ax.barh(lv["lever"], lv["macro_f1_swing"], color=cols)
for i, v in enumerate(lv["macro_f1_swing"]):
    ax.text(v + lv["macro_f1_swing"].max()*0.02, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlabel("macro F1 swing")
ax.set_title("the evaluation protocol outweighs every tuning decision combined"
             if lever.iloc[0]["lever"] == PROTO
             else "largest single lever: " + lever.iloc[0]["lever"], fontsize=11)
ax.set_xlim(0, lv["macro_f1_swing"].max()*1.3)
ax.tick_params(axis="y", labelsize=9)
plt.tight_layout()
save_fig(fig, "24_what_moves_the_result")
