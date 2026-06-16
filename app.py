# =============================================================================
#  Genel Amaçlı Veri Analizi & Makine Öğrenmesi Karar Destek Sistemi (DSS)
#  Yüklediğiniz HERHANGİ bir veri setini (CSV / SPSS .sav) uçtan uca işler:
#  veri temizleme -> EDA -> AI yorumu -> regresyon/sınıflandırma -> tahmin.
#
#  Mimari (UML): Dataset -> Preprocessor -> MLModel -> Visualizer
#  Çalıştırma:   python -m streamlit run app.py
#  Gereksinim:   pip install streamlit scikit-learn plotly pandas numpy pyreadstat
#                (opsiyonel AI: pip install google-genai)
# =============================================================================

import io
import hashlib
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, f1_score)

PALETTE = ["#e11d2a", "#0891b2", "#7c3aed", "#16a34a", "#f59e0b", "#0f172a"]


def style_fig(fig, height=320):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=13, color="#0f172a"),
        title_font=dict(family="Sora, sans-serif", size=16),
        colorway=PALETTE, margin=dict(l=10, r=10, t=46, b=10), height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def demo_data(n=300, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 65, n)
    income = rng.normal(50000, 15000, n).round()
    region = rng.choice(["Kuzey", "Güney", "Doğu", "Batı"], n)
    education = rng.choice(["Lise", "Lisans", "YüksekLisans"], n, p=[.4, .45, .15])
    edu_eff = pd.Series(education).map({"Lise": 0, "Lisans": 1, "YüksekLisans": 2}).values
    spending = 200 + 0.004 * income + 5 * age + 300 * edu_eff + rng.normal(0, 400, n)
    df = pd.DataFrame({"yas": age, "gelir": income, "bolge": region,
                       "egitim": education, "harcama": spending.round()})
    miss = rng.choice(df.index, 10, replace=False)
    df.loc[miss, "gelir"] = np.nan  # temizleme demosu için eksik
    return df


def read_uploaded(name, raw):
    if name.lower().endswith(".sav"):
        import pyreadstat
        with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as tmp:
            tmp.write(raw)
            path = tmp.name
        df, _ = pyreadstat.read_sav(path, apply_value_formats=True)
        return df
    return pd.read_csv(io.BytesIO(raw))


# =============================================================================
#  >>> SINIF 1: Dataset
# =============================================================================
class Dataset:
    def __init__(self, df=None):
        self.data = df

    def numeric_cols(self):
        return [c for c in self.data.columns if pd.api.types.is_numeric_dtype(self.data[c])]

    def categorical_cols(self):
        return [c for c in self.data.columns if c not in self.numeric_cols()]

    def summary(self):
        return {"rows": len(self.data), "cols": self.data.shape[1],
                "numeric": len(self.numeric_cols()),
                "categorical": len(self.categorical_cols()),
                "missing": int(self.data.isna().sum().sum())}
# >>> SINIF 1 SONU


# =============================================================================
#  >>> SINIF 2: Preprocessor  --- temizleme araçları + model dönüştürücüsü
# =============================================================================
class Preprocessor:
    def __init__(self, numeric=None, categorical=None):
        self.numeric = numeric or []
        self.categorical = categorical or []

    # --- İnteraktif temizleme araçları (Veri Temizleme Portalı) ---
    @staticmethod
    def fill_missing(df, cols, method):
        df = df.copy()
        for c in cols:
            if df[c].isna().sum() == 0:
                continue
            if method == "mean" and pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].mean())
            elif method == "median" and pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].median())
            elif method == "mode":
                df[c] = df[c].fillna(df[c].mode().iloc[0])
        return df

    @staticmethod
    def drop_missing_rows(df, cols):
        return df.dropna(subset=cols)

    @staticmethod
    def outlier_mask(df, col):
        q1, q3 = df[col].quantile([.25, .75])
        iqr = q3 - q1
        return (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)

    @staticmethod
    def one_hot(df, cols):
        return pd.get_dummies(df, columns=cols, dtype=int)

    # --- Modelleme dönüştürücüsü (one-hot + ölçekleme tek pipeline'da) ---
    def make_transformer(self):
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("sc", StandardScaler())])
        cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                             ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        return ColumnTransformer([("num", num_pipe, self.numeric),
                                  ("cat", cat_pipe, self.categorical)])
# >>> SINIF 2 SONU


# =============================================================================
#  >>> SINIF 3: MLModel
# =============================================================================
class MLModel:
    def __init__(self, model_type, task, pre):
        self.model_type, self.task = model_type, task
        if task == "regression":
            est = (LinearRegression() if model_type == "linear"
                   else RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
        else:
            est = (LogisticRegression(max_iter=1000) if model_type == "linear"
                   else RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
        self.pipe = Pipeline([("pre", pre.make_transformer()), ("est", est)])
        self.metrics, self.eval = {}, None

    def train(self, X_tr, y_tr, X_te, y_te):
        self.pipe.fit(X_tr, y_tr)
        pred = self.pipe.predict(X_te)
        if self.task == "regression":
            self.metrics = {"R²": r2_score(y_te, pred),
                            "MAE": mean_absolute_error(y_te, pred),
                            "RMSE": float(np.sqrt(mean_squared_error(y_te, pred)))}
        else:
            self.metrics = {"Accuracy": accuracy_score(y_te, pred),
                            "F1 (weighted)": f1_score(y_te, pred, average="weighted")}
        self.eval = (np.asarray(y_te), np.asarray(pred))
        return self.metrics

    def predict(self, X):
        return self.pipe.predict(X)

    def importances(self):
        names = self.pipe.named_steps["pre"].get_feature_names_out()
        names = [n.split("__", 1)[-1] for n in names]
        est = self.pipe.named_steps["est"]
        if hasattr(est, "feature_importances_"):
            vals = est.feature_importances_
        else:
            coef = np.asarray(est.coef_)
            vals = np.abs(coef) if coef.ndim == 1 else np.abs(coef).mean(axis=0)
        m = min(len(names), len(vals))
        return (pd.DataFrame({"feature": names[:m], "importance": vals[:m]})
                  .sort_values("importance", ascending=True).tail(15))
# >>> SINIF 3 SONU


# =============================================================================
#  >>> SINIF 4: Visualizer
# =============================================================================
class Visualizer:
    @staticmethod
    def missing_bar(df):
        miss = df.isna().sum()
        miss = miss[miss > 0].sort_values()
        if miss.empty:
            return None
        fig = px.bar(x=miss.values, y=miss.index, orientation="h",
                     title="Eksik Değer Sayısı", labels={"x": "Adet", "y": ""})
        fig.update_traces(marker_color=PALETTE[0])
        return style_fig(fig, 300)

    @staticmethod
    def corr_heatmap(df, num_cols):
        corr = df[num_cols].corr().round(2)
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Korelasyon Matrisi")
        return style_fig(fig, 480)

    @staticmethod
    def univariate(df, col, is_num):
        if is_num:
            fig = px.histogram(df, x=col, nbins=30, title=f"Dağılım — {col}")
            fig.update_traces(marker_color=PALETTE[1])
        else:
            vc = df[col].value_counts().reset_index()
            vc.columns = [col, "adet"]
            fig = px.bar(vc, x=col, y="adet", title=f"Frekans — {col}")
            fig.update_traces(marker_color=PALETTE[2])
        return style_fig(fig, 360)

    @staticmethod
    def bivariate(df, x, y, x_num, y_num, color=None):
        if x_num and y_num:
            fig = px.scatter(df, x=x, y=y, color=color, opacity=0.6, title=f"{x} vs {y}")
        elif x_num != y_num:
            cat, num = (x, y) if not x_num else (y, x)
            fig = px.box(df, x=cat, y=num, color=cat, title=f"{num} — {cat} bazında")
        else:
            tmp = df.groupby([x, y]).size().reset_index(name="adet")
            fig = px.bar(tmp, x=x, y="adet", color=y, barmode="group", title=f"{x} vs {y}")
        return style_fig(fig, 420)

    @staticmethod
    def pred_scatter(y_true, y_pred):
        fig = px.scatter(x=y_true, y=y_pred, opacity=0.6,
                         labels={"x": "Gerçek", "y": "Tahmin"}, title="Gerçek vs. Tahmin (Test)")
        lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 line=dict(dash="dash", color="#0f172a"), name="İdeal"))
        return style_fig(fig, 400)
# >>> SINIF 4 SONU


# =============================================================================
#  AI YORUMU — kural tabanlı (her zaman çalışır) + opsiyonel Gemini LLM
# =============================================================================
def build_stats_text(df, num_cols, cat_cols):
    """LLM'e/raporlamaya verilecek istatistiksel özet metni."""
    parts = [f"Boyut: {df.shape[0]} satır, {df.shape[1]} sütun. "
             f"Sayısal: {num_cols}. Kategorik: {cat_cols}.",
             f"Eksik değer toplamı: {int(df.isna().sum().sum())}."]
    if len(num_cols) >= 2:
        pairs = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                r = df[num_cols[i]].corr(df[num_cols[j]])
                if pd.notna(r):
                    pairs.append((num_cols[i], num_cols[j], r))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        parts.append("En güçlü korelasyonlar: " +
                     "; ".join(f"{a}-{b}={r:.2f}" for a, b, r in pairs[:3]))
    for c in num_cols[:4]:
        parts.append(f"{c}: ort={df[c].mean():.1f}, min={df[c].min():.1f}, max={df[c].max():.1f}")
    for c in cat_cols[:3]:
        vc = df[c].value_counts()
        if len(vc):
            parts.append(f"{c}: en sık '{vc.index[0]}' (%{100*vc.iloc[0]/len(df):.0f})")
    return "\n".join(parts)


def rule_based_report(df, num_cols, cat_cols):
    """Anahtar gerektirmeyen, istatistikten türetilen Türkçe yorum."""
    L = [f"Veri seti **{df.shape[0]} satır** ve **{df.shape[1]} sütun**dan oluşuyor "
         f"({len(num_cols)} sayısal, {len(cat_cols)} kategorik)."]
    miss = int(df.isna().sum().sum())
    L.append(f"Toplam **{miss} eksik değer** var; temizleme adımı önerilir." if miss
             else "Veride **eksik değer bulunmuyor**, modellemeye uygun.")
    if len(num_cols) >= 2:
        pairs = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                r = df[num_cols[i]].corr(df[num_cols[j]])
                if pd.notna(r):
                    pairs.append((num_cols[i], num_cols[j], r))
        if pairs:
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            a, b, r = pairs[0]
            yon = "pozitif" if r > 0 else "negatif"
            güç = "güçlü" if abs(r) >= 0.6 else ("orta düzey" if abs(r) >= 0.3 else "zayıf")
            L.append(f"En belirgin ilişki **{a}** ile **{b}** arasında: {güç} {yon} "
                     f"korelasyon (r = {r:.2f}). Yani biri artarken diğeri "
                     f"{'artma' if r>0 else 'azalma'} eğiliminde.")
    for c in cat_cols[:2]:
        vc = df[c].value_counts()
        if len(vc):
            L.append(f"**{c}** sütununda en yaygın kategori **{vc.index[0]}** "
                     f"(%{100*vc.iloc[0]/len(df):.0f}).")
    for c in num_cols[:2]:
        L.append(f"**{c}**: ortalama {df[c].mean():.1f}, aralık {df[c].min():.1f}–{df[c].max():.1f}.")
    return "\n\n".join("• " + x for x in L)


def gemini_report(api_key, model, stats_text):
    """Opsiyonel: Gemini API ile zengin Türkçe analiz metni."""
    from google import genai
    client = genai.Client(api_key=api_key)
    prompt = ("Aşağıdaki veri seti istatistiklerini bir veri analisti gibi, "
              "Türkçe, kısa paragraflar ve madde işaretleriyle yorumla. "
              "Korelasyonlara, dağılımlara ve dikkat çeken bulgulara odaklan:\n\n" + stats_text)
    resp = client.models.generate_content(model=model, contents=prompt)
    return resp.text


# =============================================================================
#  Yardımcılar (cache'li)
# =============================================================================
@st.cache_data(show_spinner=False)
def load_raw(source_key, raw=None, fname=None):
    df = demo_data() if source_key == "demo" else read_uploaded(fname, raw)
    return df.dropna(axis=1, how="all")


def detect_task(series):
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"
    return "classification" if series.nunique() <= 10 else "regression"


@st.cache_resource(show_spinner=True)
def train_models(source_key, data_version, target, features_t, task, _df):
    features = list(features_t)
    df = _df.dropna(subset=[target]).copy()
    X, y = df[features], df[target]
    num = [c for c in features if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in features if c not in num]
    pre = Preprocessor(num, cat)
    strat = y if (task == "classification" and y.nunique() > 1) else None
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
    except Exception:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {}
    for label, mtype in [("Linear/Logistic", "linear"), ("Random Forest", "random_forest")]:
        m = MLModel(mtype, task, pre)
        m.train(X_tr, y_tr, X_te, y_te)
        models[label] = m
    return models, num, cat


# =============================================================================
#  STREAMLIT ARAYÜZÜ
# =============================================================================
st.set_page_config(page_title="Genel Veri Analizi DSS", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Sora:wght@600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Sora', sans-serif; letter-spacing: -0.5px; }
[data-testid="stMetric"] { background:#fff; border:1px solid #eef0f3; border-left:4px solid #e11d2a;
  padding:14px 16px; border-radius:12px; box-shadow:0 1px 4px rgba(15,23,42,0.06); }
[data-testid="stMetricValue"] { font-family:'Sora', sans-serif; }
.app-header { background:linear-gradient(100deg,#0f172a 0%,#1e293b 60%,#7f1d1d 100%);
  color:#fff; padding:22px 26px; border-radius:16px; margin-bottom:18px; }
.app-header h1 { color:#fff; margin:0; font-size:25px; }
.app-header p { color:#cbd5e1; margin:6px 0 0; font-size:14px; }
.brand { display:flex; align-items:center; gap:8px; font-family:'Sora'; font-weight:700;
  font-size:18px; color:#e11d2a; }
</style>
""", unsafe_allow_html=True)

# ---- Sidebar ----
st.sidebar.markdown('<div class="brand">📊 Veri Analizi DSS</div>', unsafe_allow_html=True)
st.sidebar.markdown("**Genel Amaçlı Analiz & ML Sistemi**")
st.sidebar.markdown("---")

up = st.sidebar.file_uploader("📁 Veri Yükle (CSV veya .sav)", type=["csv", "sav"])
source_key, raw, fname = "demo", None, None
if up is not None:
    raw = up.getvalue()
    fname = up.name
    source_key = "u_" + hashlib.md5(raw).hexdigest()[:8]

page = st.sidebar.radio("Sayfa",
                        ["🏠 Genel Bakış", "🧹 Veri Temizleme", "🔎 EDA", "🤖 Modelleme & Tahmin"])

with st.sidebar.expander("🤖 AI Ayarları (opsiyonel)"):
    st.caption("Boş bırakırsan kural-tabanlı yorum kullanılır (anahtar gerekmez).")
    st.session_state.setdefault("gemini_key", "")
    st.session_state.setdefault("gemini_model", "gemini-2.5-flash")
    st.session_state.gemini_key = st.text_input("Gemini API Key", type="password",
                                                 value=st.session_state.gemini_key)
    st.session_state.gemini_model = st.text_input("Model", value=st.session_state.gemini_model)

st.sidebar.markdown("---")
st.sidebar.caption("Mezuniyet Projesi • Python + Streamlit + scikit-learn")
if up is None:
    st.sidebar.info("Dosya yüklenmedi — demo veri kullanılıyor.")

# ---- Veri yükle + oturum durumu (temizlenen veri kalıcı olsun) ----
try:
    raw_df = load_raw(source_key, raw, fname)
except ImportError:
    st.error("`.sav` dosyaları için `pyreadstat` gerekli: `pip install pyreadstat`")
    st.stop()
except Exception as ex:
    st.error(f"Dosya okunamadı: {ex}")
    st.stop()

if st.session_state.get("src_key") != source_key:
    st.session_state.src_key = source_key
    st.session_state.df = raw_df.copy()
    st.session_state.orig = raw_df.copy()
    st.session_state.dver = 0

df = st.session_state.df
ds = Dataset(df)
num_cols, cat_cols = ds.numeric_cols(), ds.categorical_cols()
viz = Visualizer()

st.markdown("""
<div class="app-header">
  <h1>Genel Amaçlı Veri Analizi & Makine Öğrenmesi DSS</h1>
  <p>Uçtan uca hat: veri temizleme → görselleştirme → AI yorumu → modelleme → tahmin.</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
#  SAYFA 1 — GENEL BAKIŞ (+ AI Yorumu)
# -----------------------------------------------------------------------------
if page.startswith("🏠"):
    s = ds.summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Satır", f"{s['rows']:,}")
    c2.metric("Sütun", s["cols"])
    c3.metric("Sayısal / Kategorik", f"{s['numeric']} / {s['categorical']}")
    c4.metric("Eksik Değer", f"{s['missing']:,}")

    st.markdown("### Veri Önizleme")
    st.dataframe(df.head(15), use_container_width=True)

    st.markdown("### 🤖 AI Yorumu")
    st.caption("Verinin istatistiksel özetini otomatik metin rapora dönüştürür.")
    if st.button("AI Yorumu Oluştur", type="primary"):
        stats = build_stats_text(df, num_cols, cat_cols)
        key, model = st.session_state.gemini_key, st.session_state.gemini_model
        if key:
            try:
                with st.spinner("Gemini ile rapor üretiliyor..."):
                    out = gemini_report(key, model, stats)
                st.markdown(out)
                st.caption("Kaynak: Gemini API")
            except Exception as e:
                st.warning(f"Gemini kullanılamadı ({e}). Kural-tabanlı yoruma geçildi.")
                st.markdown(rule_based_report(df, num_cols, cat_cols))
        else:
            st.markdown(rule_based_report(df, num_cols, cat_cols))
            st.caption("Kaynak: Kural-tabanlı motor (anahtarsız)")

    if num_cols:
        st.markdown("### İstatistiksel Özet (Sayısal)")
        st.dataframe(df[num_cols].describe().round(2), use_container_width=True)


# -----------------------------------------------------------------------------
#  SAYFA 2 — VERİ TEMİZLEME PORTALI
# -----------------------------------------------------------------------------
elif page.startswith("🧹"):
    st.markdown("### Veri Temizleme Portalı")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Satır", f"{len(df):,}")
    cc2.metric("Sütun", df.shape[1])
    cc3.metric("Eksik Değer", f"{int(df.isna().sum().sum()):,}")

    fig_miss = viz.missing_bar(df)
    if fig_miss:
        st.plotly_chart(fig_miss, use_container_width=True)
    else:
        st.success("Şu anda eksik değer yok ✓")

    # 1) Eksik veri
    st.markdown("#### 1) Eksik Veri")
    e1, e2, e3 = st.columns([2, 2, 1])
    miss_cols = [c for c in df.columns if df[c].isna().any()]
    target_cols = e1.multiselect("Sütun(lar)", df.columns.tolist(),
                                 default=miss_cols, key="mc")
    method = e2.selectbox("Yöntem", ["Ortalama (mean)", "Medyan (median)",
                                     "Mod (mode)", "Satırı sil"], key="mm")
    if e3.button("Uygula", key="mb"):
        if target_cols:
            if method.startswith("Satır"):
                st.session_state.df = Preprocessor.drop_missing_rows(df, target_cols)
            else:
                m = {"Ortalama (mean)": "mean", "Medyan (median)": "median",
                     "Mod (mode)": "mode"}[method]
                st.session_state.df = Preprocessor.fill_missing(df, target_cols, m)
            st.session_state.dver += 1
            st.rerun()

    # 2) Aykırı değer (IQR)
    st.markdown("#### 2) Aykırı Değer (Outlier) — IQR Yöntemi")
    if num_cols:
        o1, o2 = st.columns([3, 1])
        ocol = o1.selectbox("Sayısal sütun", num_cols, key="oc")
        n_out = int(Preprocessor.outlier_mask(df, ocol).sum())
        st.caption(f"'{ocol}' sütununda **{n_out}** aykırı değer tespit edildi.")
        if o2.button("Temizle", key="ob") and n_out > 0:
            st.session_state.df = df[~Preprocessor.outlier_mask(df, ocol)]
            st.session_state.dver += 1
            st.rerun()
    else:
        st.info("Sayısal sütun yok.")

    # 3) One-Hot Encoding
    st.markdown("#### 3) Kategorik Kodlama (One-Hot Encoding)")
    if cat_cols:
        h1, h2 = st.columns([3, 1])
        hcols = h1.multiselect("Kodlanacak kategorik sütun(lar)", cat_cols, key="hc")
        if h2.button("Kodla", key="hb") and hcols:
            st.session_state.df = Preprocessor.one_hot(df, hcols)
            st.session_state.dver += 1
            st.rerun()
    else:
        st.info("Kategorik sütun yok.")

    # 4) Diğer + sıfırlama + indirme
    st.markdown("#### 4) Diğer İşlemler")
    d1, d2, d3 = st.columns(3)
    if d1.button("Tekrarlı satırları sil"):
        st.session_state.df = df.drop_duplicates()
        st.session_state.dver += 1
        st.rerun()
    drop_cols = d2.multiselect("Sütun sil", df.columns.tolist(), key="dc")
    if d2.button("Seçili sütunları sil") and drop_cols:
        st.session_state.df = df.drop(columns=drop_cols)
        st.session_state.dver += 1
        st.rerun()
    if d3.button("↩️ Orijinale dön"):
        st.session_state.df = st.session_state.orig.copy()
        st.session_state.dver += 1
        st.rerun()

    st.markdown("#### Güncel Veri")
    st.dataframe(df.head(15), use_container_width=True)
    st.download_button("⬇️ Temizlenmiş CSV indir",
                       df.to_csv(index=False).encode("utf-8-sig"),
                       "temizlenmis_veri.csv", "text/csv")


# -----------------------------------------------------------------------------
#  SAYFA 3 — EDA
# -----------------------------------------------------------------------------
elif page.startswith("🔎"):
    if len(num_cols) >= 2:
        st.markdown("### Korelasyon")
        st.plotly_chart(viz.corr_heatmap(df, num_cols), use_container_width=True)
    else:
        st.info("Korelasyon için en az 2 sayısal sütun gerekir.")

    st.markdown("### Tek Değişken Analizi")
    col = st.selectbox("Sütun seç", list(df.columns))
    st.plotly_chart(viz.univariate(df, col, col in num_cols), use_container_width=True)

    st.markdown("### İki Değişken İlişkisi")
    b1, b2, b3 = st.columns(3)
    xcol = b1.selectbox("X", list(df.columns), index=0)
    ycol = b2.selectbox("Y", list(df.columns), index=min(1, len(df.columns) - 1))
    color = b3.selectbox("Renk (opsiyonel)", ["(yok)"] + cat_cols)
    color = None if color == "(yok)" else color
    samp = df.sample(min(3000, len(df)), random_state=1)
    st.plotly_chart(viz.bivariate(samp, xcol, ycol, xcol in num_cols, ycol in num_cols, color),
                    use_container_width=True)


# -----------------------------------------------------------------------------
#  SAYFA 4 — MODELLEME & TAHMİN
# -----------------------------------------------------------------------------
else:
    st.markdown("### 1) Hedef ve Öznitelik Seçimi")
    target = st.selectbox("🎯 Tahmin edilecek hedef sütun", list(df.columns))
    auto = detect_task(df[target])
    task_label = st.radio("Görev tipi",
                          ["Regresyon (sayısal tahmin)", "Sınıflandırma (kategori tahmini)"],
                          index=0 if auto == "regression" else 1, horizontal=True)
    task = "regression" if task_label.startswith("Reg") else "classification"
    st.caption(f"Otomatik öneri: **{'Regresyon' if auto=='regression' else 'Sınıflandırma'}** "
               f"(hedefin {df[target].nunique()} benzersiz değeri var)")

    candidates = [c for c in df.columns if c != target]
    features = st.multiselect("📥 Öznitelikler (girdi)", candidates, default=candidates)
    if not features:
        st.warning("En az bir öznitelik seç.")
        st.stop()

    models, num, cat = train_models(source_key, st.session_state.dver, target,
                                    tuple(features), task, df)
    rf = models["Random Forest"]

    st.markdown("### 2) Model Performansı")
    perf = pd.DataFrame({k: m.metrics for k, m in models.items()}).T.round(3)
    st.dataframe(perf, use_container_width=True)

    cL, cR = st.columns(2)
    if task == "regression":
        cL.plotly_chart(viz.pred_scatter(*rf.eval), use_container_width=True)
    else:
        ev = pd.DataFrame({"gerçek": rf.eval[0].astype(str), "tahmin": rf.eval[1].astype(str)})
        figcm = px.density_heatmap(ev, x="tahmin", y="gerçek", title="Karışıklık Matrisi (Test)",
                                   color_continuous_scale="Reds", text_auto=True)
        cL.plotly_chart(style_fig(figcm, 400), use_container_width=True)

    imp = rf.importances()
    fig_imp = px.bar(imp, x="importance", y="feature", orientation="h",
                     title="Random Forest — Öznitelik Önemi")
    fig_imp.update_traces(marker_color=PALETTE[3])
    cR.plotly_chart(style_fig(fig_imp, 400), use_container_width=True)

    st.markdown("---")
    st.markdown("### 3) 🎛️ Yeni Tahmin")
    with st.expander("Girdi değerleri", expanded=True):
        inputs = {}
        cols = st.columns(3)
        for i, f in enumerate(features):
            c = cols[i % 3]
            if f in num:
                inputs[f] = c.number_input(f, value=float(df[f].median()))
            else:
                opts = sorted(df[f].dropna().astype(str).unique().tolist())
                inputs[f] = c.selectbox(f, opts)
    if st.button("🔮 Tahmin Et", type="primary", use_container_width=True):
        row = pd.DataFrame([inputs])[features]
        pred = rf.predict(row)[0]
        st.metric(f"Tahmin: {target}", f"{pred:,.2f}" if task == "regression" else f"{pred}")
        st.success("Tahmin Random Forest modeliyle üretildi.")
