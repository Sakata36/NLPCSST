import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from math import erfc, sqrt, log, exp

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

import datetime
import re

def pearsonr(x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    if n < 3 or abs(r) == 1.0:
        return r, 0.0
    t = r * sqrt(n - 2) / sqrt(1 - r ** 2)
    p = 2 * (1 - 0.5 * erfc(-abs(t) / sqrt(2)))
    return r, p

class TextBlob:
    _analyzer = None

    def __init__(self, text):
        self.text = text
        if TextBlob._analyzer is None:
            TextBlob._analyzer = SentimentIntensityAnalyzer()
        scores = TextBlob._analyzer.polarity_scores(text)
        self._polarity = scores['compound']

        self._subjectivity = abs(scores['compound']) * 0.6 + scores['pos'] * 0.2 + scores['neg'] * 0.2

    @property
    def sentiment(self):
        class _Sentiment:
            def __init__(self, pol, sub):
                self.polarity = pol
                self.subjectivity = sub
        return _Sentiment(self._polarity, self._subjectivity)

# SYNTHETIC DATASET — SIMULATED SOCIAL MEDIA POSTS & RICE PRICES

SOCIAL_MEDIA_DATA = [
    # (date_str, post_text, rice_price_php_per_kg)
    # ── 2023 ──
    ("2023-01-15", "Bigas pa rin ang pinaka-affordable na pagkain namin. Salamat sa presyo ngayon!", 45.0),
    ("2023-01-22", "Rice is still manageable in price. We can still afford it thankfully.", 45.2),
    ("2023-02-10", "Medyo tumaas ang bigas sa aming palengke pero okay pa rin.", 46.5),
    ("2023-02-18", "Why is rice getting more expensive? The market price increased this month.", 46.8),
    ("2023-03-05", "Maganda pa rin ang ani this season. Rice prices remain stable so far.", 46.0),
    ("2023-03-20", "Rice supply seems okay in our area. No major price hike noticed.", 45.8),
    ("2023-04-07", "Nagsimula nang tumaas ang presyo ng bigas. Nakakaalarma na ito.", 47.5),
    ("2023-04-19", "Rice prices are rising! This is getting difficult for poor families.", 48.0),
    ("2023-05-03", "Ang mahal na ng bigas! Hindi na kaya ng ordinaryong Pilipino ito.", 49.5),
    ("2023-05-14", "The price of rice has increased dramatically. This is unacceptable!", 50.2),
    ("2023-06-01", "Grabe ang pagtaas ng bigas. P55 na per kilo sa aming lugar!", 54.5),
    ("2023-06-15", "Rice hitting new highs. Families are struggling to buy enough food.", 55.3),
    ("2023-07-04", "El Niño effect hitting us hard. Rice supply is very limited now.", 56.8),
    ("2023-07-18", "Walang bigas sa palengke! Very long queue at NFA outlets today.", 57.2),
    ("2023-08-01", "Government must act now! Rice prices are destroying family budgets.", 58.5),
    ("2023-08-12", "Taas presyo ng bigas hindi pa humihinto. Nakakainis at nakakalungkot.", 59.0),
    ("2023-09-05", "Some relief expected as government intervenes in rice price control.", 55.0),
    ("2023-09-20", "Slight drop in rice prices noticed at the local market. Hopefully sustained.", 54.2),
    ("2023-10-08", "Importasyon ng bigas ay nakatulong sa pagbaba ng presyo. Salamat!", 52.0),
    ("2023-10-22", "Rice prices coming down slowly. Families feel slightly better now.", 51.5),
    ("2023-11-05", "Harvest season helped stabilize rice prices. Affordable again!", 49.8),
    ("2023-11-19", "Bigas na mas mura ngayon. Maayos na ang supply chain dito sa amin.", 49.0),
    ("2023-12-03", "Good news – rice prices dropped significantly this month. Very happy!", 47.5),
    ("2023-12-20", "Pasasalamat sa pagbaba ng bigas. Mas kaya na ng aming pamilya.", 47.0),
    ("2023-01-28", "Bigas ay still affordable ngayon. No complaints from our market.", 45.1),
    ("2023-03-12", "Rice supply is normal this March. Farmers are optimistic about harvest.", 45.9),
    ("2023-05-25", "Prices keep climbing. Mahirap na talaga bumili ng bigas ngayon.", 50.8),
    ("2023-07-25", "No rice at the store again today! This shortage is really alarming.", 57.5),
    ("2023-09-14", "Government price caps on rice are helping. Slight relief felt today.", 54.6),
    ("2023-11-28", "Thanksgiving vibes — bigas is becoming affordable again. Grateful!", 49.2),
    # ── 2024 ──
    ("2024-01-08", "New year, better rice prices! Still manageable for most families.", 46.5),
    ("2024-01-20", "Rice is still at a reasonable price. Hoping it stays this way.", 46.8),
    ("2024-02-05", "Another increase in rice price observed. Not good for poor communities.", 48.0),
    ("2024-02-17", "Prices creeping up again. Is another shortage coming? Very worried.", 48.5),
    ("2024-03-03", "Rice market feels unstable. Prices fluctuating week by week.", 49.0),
    ("2024-03-18", "Supply issues causing uncertainty. Market prices going up and down.", 49.2),
    ("2024-04-07", "Summer na. Nagsimula ulit na tumaas ang presyo ng bigas.", 50.5),
    ("2024-04-21", "Prices rising again in April. Farmers say low yield this dry season.", 51.0),
    ("2024-05-06", "Very difficult to buy enough rice this month. Prices are too high.", 52.5),
    ("2024-05-19", "Mga pamilyang mahirap ay apektado ng taas presyo ng bigas ngayon.", 53.0),
    ("2024-06-02", "Rainy season should help improve harvest. Some hope for price drop.", 52.0),
    ("2024-06-16", "Government rice programs helping some communities. Mixed results though.", 51.5),
    # ── 2024 H2 ──
    ("2024-07-05", "Bumaba na ng konti ang bigas. Maayos na ang supply sa palengke.", 50.5),
    ("2024-07-19", "Rice prices are slowly dropping this July. Families feeling a bit of relief.", 50.0),
    ("2024-08-03", "Harvest season coming. Farmers optimistic about rice supply this year.", 49.0),
    ("2024-08-17", "Presyo ng bigas ay patuloy na bumababa. Salamat sa magandang ani!", 48.5),
    ("2024-09-07", "Good harvest this rainy season! Rice is now more affordable at P48/kg.", 48.0),
    ("2024-09-21", "Supply is stable now. Bigas ay mabibili na sa makatwirang presyo.", 47.8),
    ("2024-10-05", "Rice prices slightly up again due to typhoon damage in rice-growing areas.", 49.5),
    ("2024-10-20", "Typhoon nakapinsala sa mga sakahan. Takot na kami sa pagtaas ng bigas.", 50.0),
    ("2024-11-04", "Rice supply slowly recovering post-typhoon. Prices stabilizing this month.", 49.2),
    ("2024-11-18", "Bigas ay abot-kaya na naman. Masaya ang mga pamilya sa pagbaba ng presyo.", 48.0),
    ("2024-12-06", "Holiday season and rice prices remain manageable. Great news for families!", 47.5),
    ("2024-12-22", "Pasko na at ang bigas ay hindi masyadong mahal. Masaya kaming pamilya.", 47.0),
    ("2024-01-29", "Rice prices holding steady this month. Cautiously optimistic.", 46.6),
    ("2024-03-25", "Market is uncertain. Presyo ng bigas fluctuating every week here.", 49.1),
    ("2024-05-28", "Community kitchen struggling because of high rice prices right now.", 52.8),
    ("2024-08-25", "Post-harvest supply improving. Bigas prices dropping nicely.", 48.2),
    ("2024-10-28", "Recovery from typhoon damage is slow. Rice prices still unstable.", 49.8),
    ("2024-12-14", "Good news before Christmas — bigas prices down to affordable range!", 47.2),
    # ── 2025 ──
    ("2025-01-10", "New year brings stable rice prices. Government monitoring supply closely.", 46.5),
    ("2025-01-24", "Rice still affordable in January. Community is hopeful for continued stability.", 46.8),
    ("2025-02-08", "Bigas ay medyo tumaas ngayong Pebrero. Sana ay hindi na tumaas pa.", 48.0),
    ("2025-02-20", "Price increase noticed again. Worried this will affect our daily budget.", 48.5),
    ("2025-03-07", "Dry season worries again. Farmers concerned about irrigation and yield.", 49.5),
    ("2025-03-21", "El Nino threat resurfaces. Rice prices may rise again in coming months.", 50.0),
    ("2025-04-05", "Presyo ng bigas ay patuloy na lumolobog. Nakakaapekto sa aming buhay.", 51.5),
    ("2025-04-18", "Rising prices again this summer. Very hard for low-income families here.", 52.0),
    ("2025-05-03", "Government announces rice importation plan. Some hope for price reduction soon.", 52.5),
    ("2025-05-17", "Balita ng importasyon ng bigas ay nagbigay ng pag-asa sa aming komunidad.", 52.0),
    ("2025-06-02", "First shipments of imported rice arriving. Prices beginning to ease slightly.", 51.0),
    ("2025-06-16", "Rice prices slowly dropping as imported supply hits local markets. Relief!", 50.5),
    ("2025-07-04", "Good news -- rice now at P49/kg. Supply is improving significantly.", 49.0),
    ("2025-07-19", "Bigas ay mas mura na ngayon sa palengke. Maganda ang epekto ng importasyon.", 48.5),
    ("2025-08-02", "Harvest season boosting local supply. Community happy with stable prices.", 47.5),
    ("2025-08-16", "Rice prices are stable and affordable. Farmers reporting good yields.", 47.0),
    ("2025-09-06", "Stable bigas supply this September. Walang shortage na nararamdaman.", 46.5),
    ("2025-09-20", "Community satisfied with current rice prices. Hoping it lasts long-term.", 46.2),
    ("2025-10-04", "Minor price increase due to logistics issues. Still manageable overall.", 47.5),
    ("2025-10-19", "Slight uptick in rice price but not alarming. Supply remains adequate.", 47.8),
    ("2025-11-03", "Harvest yields excellent this year! Rice prices dropping further.", 46.0),
    ("2025-11-17", "Magandang ani ngayong Nobyembre. Bigas ay mas mura na ng kaunti.", 45.5),
    ("2025-12-05", "Year-end prices stable. Families can afford rice for the holidays!", 45.0),
    ("2025-12-20", "Masayang Pasko! Abot-kaya ang bigas para sa aming pamilya ngayong kapaskuhan.", 45.0),
    ("2025-02-14", "Valentine's Day and rice prices are giving no love — going up again.", 48.2),
    ("2025-04-25", "Summer heat is brutal. Rice yields expected to drop this season.", 51.8),
    ("2025-06-28", "Importation efforts paying off. Presyo ng bigas is going down!", 50.8),
    ("2025-09-14", "Harvest festivals and affordable rice — community is happy!", 46.4),
    ("2025-11-28", "Year-end harvest surplus keeping prices low. Great for families.", 45.8),
    # ── 2026 ──
    ("2026-01-09", "New year 2026 -- rice prices remain low and stable. Very positive outlook.", 44.5),
    ("2026-01-23", "Community groups report improved food security situation this January.", 44.8),
    ("2026-02-07", "Bigas ay mura pa rin ngayong simula ng taon. Maganda ang sitwasyon.", 45.0),
    ("2026-02-21", "Prices holding steady. Government supply chain improvements working well.", 45.2),
    ("2026-03-07", "Early dry season but supply reserves strong. No shortage expected soon.", 46.0),
    ("2026-03-21", "Rice prices stable despite dry season. Buffer stocks keeping markets calm.", 46.3),
    ("2026-04-05", "Summer heat affecting crops but prices manageable. Government acting fast.", 47.0),
    ("2026-04-19", "Slight increase in rice price this April. Market still within acceptable range.", 47.5),
    ("2026-05-03", "Community concerned about rising prices heading into mid-year. Watch closely.", 48.0),
    ("2026-05-17", "Rice price at P48 -- higher than last year but better than 2023 crisis.", 48.2),
    ("2026-03-28", "Dry season concerns but government buffer stocks keeping prices stable.", 46.2),
]

# NLP SENTIMENT ANALYSIS MODULE

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s!?.,'áéíóúàèìòùñ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def analyze_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores['compound'], scores['pos'], scores['neg'], scores['neu']

def analyze_sentiment_textblob(text):

    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def classify_sentiment(compound_score):

    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def perform_nlp_analysis(data):

    print("\n" + "="*70)
    print("  NLP SENTIMENT ANALYSIS MODULE")
    print("="*70)
    print(f"  Processing {len(data)} social media posts...\n")

    records = []
    for date_str, post, price in data:
        cleaned = clean_text(post)
        vader_compound, vader_pos, vader_neg, vader_neu = analyze_sentiment_vader(post)
        blob_polarity, blob_subjectivity = analyze_sentiment_textblob(post)


        ensemble_score = (0.7 * vader_compound) + (0.3 * blob_polarity)
        label = classify_sentiment(vader_compound)

        records.append({
            "date": pd.to_datetime(date_str),
            "post": post,
            "cleaned_post": cleaned,
            "rice_price": price,
            "vader_compound": round(vader_compound, 4),
            "vader_positive": round(vader_pos, 4),
            "vader_negative": round(vader_neg, 4),
            "vader_neutral": round(vader_neu, 4),
            "textblob_polarity": round(blob_polarity, 4),
            "textblob_subjectivity": round(blob_subjectivity, 4),
            "ensemble_score": round(ensemble_score, 4),
            "sentiment_label": label,
        })

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    print("  Sample Sentiment Results:")
    print(f"  {'Date':<12} {'VADER':>8} {'TextBlob':>10} {'Label':<12} {'Price':>8}")
    print("  " + "-"*55)
    for _, row in df.iterrows():
        print(f"  {str(row['date'].date()):<12} "
              f"{row['vader_compound']:>8.3f} "
              f"{row['textblob_polarity']:>10.3f} "
              f"{row['sentiment_label']:<12} "
              f"₱{row['rice_price']:>6.1f}")

    print(f"\n  Sentiment Distribution:")
    label_counts = df['sentiment_label'].value_counts()
    for label, count in label_counts.items():
        pct = count / len(df) * 100
        print(f"    {label}: {count} posts ({pct:.1f}%)")

    corr, pval = pearsonr(df['vader_compound'], df['rice_price'])
    print(f"\n  NLP-Price Correlation (Pearson r): {corr:.4f}  (p-value: {pval:.4f})")
    print(f"  Interpretation: {'Significant negative correlation' if corr < -0.3 else 'Moderate correlation'}")

    return df

def aggregate_monthly(df):

    df['month'] = df['date'].dt.to_period('M')
    monthly = df.groupby('month').agg(
        rice_price=('rice_price', 'mean'),
        sentiment_score=('vader_compound', 'mean'),
        post_count=('post', 'count'),
        positive_pct=('sentiment_label', lambda x: (x == 'Positive').mean() * 100),
        negative_pct=('sentiment_label', lambda x: (x == 'Negative').mean() * 100),
    ).reset_index()
    monthly['month_dt'] = monthly['month'].dt.to_timestamp()
    return monthly

def check_stationarity(series, name="Series"):

    result = adfuller(series.dropna())
    is_stationary = result[1] < 0.05
    print(f"\n  ADF Test — {name}:")
    print(f"    ADF Statistic : {result[0]:.4f}")
    print(f"    p-value       : {result[1]:.4f}")
    print(f"    Stationary    : {'Yes ✓' if is_stationary else 'No — differencing needed'}")
    return is_stationary

def fit_arima_model(monthly_df, forecast_steps=6):
    print("\n" + "="*70)
    print("  COMPUTATIONAL SCIENCE — ARIMA FORECASTING MODULE")
    print("="*70)

    prices = monthly_df['rice_price'].values
    sentiment = monthly_df['sentiment_score'].values

    check_stationarity(monthly_df['rice_price'], "Rice Price Series")

    train_size = len(prices) - 3
    train_price = prices[:train_size]
    test_price = prices[train_size:]
    train_sentiment = sentiment[:train_size]
    test_sentiment = sentiment[train_size:]

    print(f"\n  Train set size : {train_size} months")
    print(f"  Test set size  : {len(test_price)} months")

    print("\n  Fitting ARIMA(1,1,1) model with sentiment as exogenous variable...")
    model = ARIMA(train_price, exog=train_sentiment.reshape(-1, 1), order=(1, 1, 1))
    fitted = model.fit()
    print(f"\n  Model AIC : {fitted.aic:.2f}")
    print(f"  Model BIC : {fitted.bic:.2f}")
    print(f"  Log-Likelihood: {fitted.llf:.2f}")

    train_pred = fitted.fittedvalues

    future_sentiment = np.linspace(
        sentiment[-1], sentiment[-1] * 0.8, forecast_steps
    )
    all_future_exog = np.concatenate([test_sentiment, future_sentiment]).reshape(-1, 1)

    forecast_result = fitted.get_forecast(
        steps=len(test_price) + forecast_steps,
        exog=all_future_exog
    )
    all_forecast = forecast_result.predicted_mean
    conf_int_df = forecast_result.conf_int(alpha=0.05)  # 95% CI (DataFrame)

    test_forecast = all_forecast[:len(test_price)]
    future_forecast = all_forecast[len(test_price):]

    mae = mean_absolute_error(test_price, test_forecast)
    rmse = np.sqrt(mean_squared_error(test_price, test_forecast))
    mape = np.mean(np.abs((test_price - test_forecast) / test_price)) * 100

    print(f"\n  Model Evaluation (Test Set):")
    print(f"    MAE  (Mean Absolute Error)      : ₱{mae:.2f}/kg")
    print(f"    RMSE (Root Mean Square Error)   : ₱{rmse:.2f}/kg")
    print(f"    MAPE (Mean Abs Percentage Error): {mape:.2f}%")

    last_date = monthly_df['month_dt'].iloc[-1]
    future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(forecast_steps)]

    ci_arr = np.array(conf_int_df)  # ensure numpy array shape (n_steps, 2)

    print(f"\n  6-Month Price Forecast (₱/kg):")
    print(f"  {'Month':<15} {'Forecast':>10} {'Lower 95% CI':>14} {'Upper 95% CI':>14}")
    print("  " + "-"*55)
    for i, (fdate, fval) in enumerate(zip(future_dates, future_forecast)):
        lo = ci_arr[len(test_price) + i, 0]
        hi = ci_arr[len(test_price) + i, 1]
        print(f"  {str(fdate.date()):<15} ₱{fval:>8.2f}  [{lo:>8.2f}, {hi:>8.2f}]")

    return {
        "monthly_df": monthly_df,
        "train_size": train_size,
        "train_pred": train_pred,
        "test_price": test_price,
        "test_forecast": test_forecast,
        "future_forecast": future_forecast,
        "future_dates": future_dates,
        "ci_arr": ci_arr,
        "mae": mae, "rmse": rmse, "mape": mape,
        "fitted_model": fitted,
        "prices": prices,
        "sentiment": sentiment,
    }

# VISUALIZATION MODULE
#   Generates 4 publication-quality charts:
#   1. Sentiment Trend Over Time
#   2. Sentiment Distribution (Pie/Bar)
#   3. Rice Price Actual vs ARIMA Forecast
#   4. NLP-Price Correlation Scatter Plot

PALETTE = {
    "bg": "#0F172A",
    "panel": "#1E293B",
    "accent1": "#38BDF8",   # sky blue
    "accent2": "#F97316",   # orange
    "accent3": "#4ADE80",   # green
    "accent4": "#F43F5E",   # rose
    "positive": "#4ADE80",
    "neutral": "#FACC15",
    "negative": "#F43F5E",
    "text": "#E2E8F0",
    "muted": "#94A3B8",
    "grid": "#334155",
}

def style_axis(ax):

    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])
    ax.grid(color=PALETTE["grid"], linewidth=0.5, alpha=0.7)

def create_visualizations(df_raw, arima_results):

    monthly = arima_results["monthly_df"]
    train_size = arima_results["train_size"]
    prices = arima_results["prices"]
    sentiment = arima_results["sentiment"]
    future_dates = arima_results["future_dates"]
    future_forecast = arima_results["future_forecast"]
    ci_arr = arima_results["ci_arr"]
    test_price = arima_results["test_price"]
    test_forecast = arima_results["test_forecast"]

    fig = plt.figure(figsize=(18, 12), facecolor=PALETTE["bg"])
    fig.suptitle(
        "Philippine Rice Price Sentiment & ARIMA Forecasting System\n"
        "NLP + Computational Science Integration",
        color=PALETTE["text"], fontsize=16, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32,
                           left=0.07, right=0.97, top=0.92, bottom=0.07)

    dates = df_raw['date'].values
    vader_scores = df_raw['vader_compound'].values
    sent_labels = df_raw['sentiment_label'].values

    # Sentiment Trend Over Time
    ax1 = fig.add_subplot(gs[0, 0])
    style_axis(ax1)

    colors = [PALETTE["positive"] if l == "Positive"
              else PALETTE["negative"] if l == "Negative"
              else PALETTE["neutral"] for l in sent_labels]

    ax1.scatter(dates, vader_scores, c=colors, s=60, alpha=0.85, zorder=3)

    # Rolling average line
    window = min(5, len(vader_scores))
    rolling_avg = pd.Series(vader_scores).rolling(window, center=True).mean()
    ax1.plot(dates, rolling_avg, color=PALETTE["accent1"], linewidth=2.2,
             label=f"Rolling Avg (n={window})", zorder=4)

    ax1.axhline(0, color=PALETTE["muted"], linewidth=0.8, linestyle="--", alpha=0.6)
    ax1.fill_between(dates, 0, vader_scores,
                     where=vader_scores >= 0, alpha=0.12, color=PALETTE["positive"])
    ax1.fill_between(dates, 0, vader_scores,
                     where=vader_scores < 0, alpha=0.12, color=PALETTE["negative"])

    ax1.set_title("① Sentiment Trend Over Time", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Date", fontsize=9)
    ax1.set_ylabel("VADER Compound Score", fontsize=9)
    ax1.set_ylim(-1.1, 1.1)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE["positive"],
               markersize=8, label='Positive'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE["neutral"],
               markersize=8, label='Neutral'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE["negative"],
               markersize=8, label='Negative'),
        Line2D([0], [0], color=PALETTE["accent1"], linewidth=2, label=f'Rolling Avg'),
    ]
    ax1.legend(handles=legend_elements, loc='lower left', fontsize=7,
               facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
               labelcolor=PALETTE["text"])

    # Sentiment Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    style_axis(ax2)

    label_counts = df_raw['sentiment_label'].value_counts()
    categories = ["Positive", "Neutral", "Negative"]
    counts = [label_counts.get(c, 0) for c in categories]
    bar_colors = [PALETTE["positive"], PALETTE["neutral"], PALETTE["negative"]]

    bars = ax2.bar(categories, counts, color=bar_colors, width=0.55,
                   edgecolor=PALETTE["bg"], linewidth=1.5, zorder=3)
    for bar, count in zip(bars, counts):
        pct = count / sum(counts) * 100
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{count}\n({pct:.0f}%)",
                 ha='center', va='bottom', color=PALETTE["text"],
                 fontsize=9, fontweight='bold')

    ax2.set_title("② Sentiment Distribution", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("Sentiment Label", fontsize=9)
    ax2.set_ylabel("Number of Posts", fontsize=9)
    ax2.set_ylim(0, max(counts) * 1.3)
    ax2.grid(axis='x', alpha=0)

    # Rice Price — Actual vs ARIMA Forecast
    ax3 = fig.add_subplot(gs[1, 0])
    style_axis(ax3)

    month_dates = monthly['month_dt'].values
    n_months = len(month_dates)

    ax3.plot(month_dates, prices, color=PALETTE["accent1"], linewidth=2.2,
             marker='o', markersize=5, label="Actual Price", zorder=4)

    train_dates = month_dates[:train_size]
    test_dates = month_dates[train_size:]
    train_pred = arima_results["train_pred"]

    ax3.plot(train_dates, train_pred,
             color=PALETTE["accent3"], linewidth=1.5, linestyle="--",
             label="ARIMA Fitted (Train)", alpha=0.85, zorder=3)

    ax3.plot(test_dates, test_forecast,
             color=PALETTE["accent2"], linewidth=1.8, linestyle="--",
             marker='s', markersize=5, label="ARIMA Test Forecast", zorder=4)

    future_date_arr = np.array([d for d in future_dates])
    ax3.plot(future_date_arr, future_forecast,
             color=PALETTE["accent4"], linewidth=2.2, linestyle="-",
             marker='^', markersize=6, label="6-Month Forecast", zorder=5)

    ci_lower = ci_arr[len(test_price):, 0]
    ci_upper = ci_arr[len(test_price):, 1]
    ax3.fill_between(future_date_arr, ci_lower, ci_upper,
                     alpha=0.20, color=PALETTE["accent4"], label="95% CI")

    split_date = month_dates[train_size - 1]
    ax3.axvline(pd.Timestamp(split_date), color=PALETTE["muted"],
                linewidth=1, linestyle=':', alpha=0.7)
    ax3.text(pd.Timestamp(split_date), prices.max() + 0.5, "Train|Test",
             color=PALETTE["muted"], fontsize=7, ha='center')

    ax3.set_title(
        f"③ Rice Price: Actual vs ARIMA Forecast  "
        f"(MAE=₱{arima_results['mae']:.2f}, MAPE={arima_results['mape']:.1f}%)",
        fontsize=10, fontweight="bold", pad=10
    )
    ax3.set_xlabel("Month", fontsize=9)
    ax3.set_ylabel("Price (₱/kg)", fontsize=9)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax3.legend(loc='upper left', fontsize=7, facecolor=PALETTE["panel"],
               edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])

    # NLP Sentiment vs Rice Price Correlation
    ax4 = fig.add_subplot(gs[1, 1])
    style_axis(ax4)

    corr, pval = pearsonr(df_raw['vader_compound'].values,
                          df_raw['rice_price'].values)

    scatter_colors = [PALETTE["positive"] if l == "Positive"
                      else PALETTE["negative"] if l == "Negative"
                      else PALETTE["neutral"]
                      for l in df_raw['sentiment_label']]

    ax4.scatter(df_raw['vader_compound'].values, df_raw['rice_price'].values,
                c=scatter_colors, s=65, alpha=0.85, zorder=3,
                edgecolors=PALETTE["bg"], linewidths=0.5)

    m, b = np.polyfit(df_raw['vader_compound'].values,
                      df_raw['rice_price'].values, 1)
    x_line = np.linspace(df_raw['vader_compound'].min(),
                         df_raw['vader_compound'].max(), 100)
    ax4.plot(x_line, m * x_line + b, color=PALETTE["accent1"],
             linewidth=2, alpha=0.9, label=f"Linear Fit (r={corr:.2f})")

    ax4.set_title("④ NLP Sentiment vs Rice Price Correlation",
                  fontsize=11, fontweight="bold", pad=10)
    ax4.set_xlabel("VADER Compound Sentiment Score", fontsize=9)
    ax4.set_ylabel("Rice Price (₱/kg)", fontsize=9)
    ax4.text(0.05, 0.92,
             f"Pearson r = {corr:.3f}\np-value = {pval:.4f}",
             transform=ax4.transAxes, color=PALETTE["text"],
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=PALETTE["panel"],
                       edgecolor=PALETTE["grid"], alpha=0.9))
    ax4.legend(loc='lower left', fontsize=8, facecolor=PALETTE["panel"],
               edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])

    # Footer
    fig.text(0.5, 0.01,
             "NLP + CSST |  "
             "Philippine Rice Price Sentiment & ARIMA Forecasting  |  "
             "Group 5 PRESENTATION",
             ha='center', va='bottom', color=PALETTE["muted"], fontsize=8)

    output_path = "rice_sentiment_forecast_visualization.png"  # saves in same folder as script
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=PALETTE["bg"])
    plt.show()
    print(f"\n  ✔ Visualization saved → {output_path}")
    return output_path


# SYSTEM INTEGRATION & MAIN PIPELINE

def export_results_csv(df_raw, monthly, arima_results):

    output = "rice_sentiment_results.csv"  # saves in same folder as script
    df_raw.drop(columns=['cleaned_post']).to_csv(output, index=False)
    print(f"  ✔ Results CSV saved → {output}")

def main():
    print("\n" + "█"*70)
    print("  PHILIPPINE RICE PRICE SENTIMENT & ARIMA FORECASTING SYSTEM")
    print("  NLP + CSST")
    print("  Final Project | Group 5 PRESENTATION")
    print("█"*70)


    df_raw = perform_nlp_analysis(SOCIAL_MEDIA_DATA)


    monthly = aggregate_monthly(df_raw)


    arima_results = fit_arima_model(monthly, forecast_steps=6)

    print("\n" + "="*70)
    print("  VISUALIZATION MODULE")
    print("="*70)
    viz_path = create_visualizations(df_raw, arima_results)

    export_results_csv(df_raw, monthly, arima_results)

    # System Summary
    print("\n" + "="*70)
    print("  SYSTEM SUMMARY")
    print("="*70)
    print(f"  Total Posts Analyzed       : {len(df_raw)}")
    print(f"  Date Range                 : {df_raw['date'].min().date()} → {df_raw['date'].max().date()}")
    print(f"  Positive Sentiment Posts   : {(df_raw['sentiment_label'] == 'Positive').sum()}")
    print(f"  Negative Sentiment Posts   : {(df_raw['sentiment_label'] == 'Negative').sum()}")
    print(f"  Neutral Sentiment Posts    : {(df_raw['sentiment_label'] == 'Neutral').sum()}")
    print(f"  Price Range Observed       : ₱{df_raw['rice_price'].min():.1f} – ₱{df_raw['rice_price'].max():.1f}/kg")
    print(f"  ARIMA Model MAPE           : {arima_results['mape']:.2f}%")
    print(f"  6-Month Forecast Range     : ₱{arima_results['future_forecast'].min():.2f} – ₱{arima_results['future_forecast'].max():.2f}/kg")
    print(f"\n  ✔ System pipeline completed successfully.")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()