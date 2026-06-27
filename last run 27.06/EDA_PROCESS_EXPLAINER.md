# מדריך טכני מלא — תהליך הנתונים, EDA והמודלים
## פרויקט זיהוי סימפטומי PTSD בסלנג צבאי עברי (Project Sasha)

---

## תוכן עניינים

1. [תמונת על — Pipeline](#1-תמונת-על--pipeline)
2. [קובץ הנתונים — dataset.json](#2-קובץ-הנתונים--datasetjson)
3. [Stage 3 — eda.py (EDA + בייסליין TF-IDF/LR)](#3-stage-3--edapy-eda--בייסליין-tf-idflr)
4. [Stage 3.5 — validation.py](#4-stage-35--validationpy)
5. [Stage 4 — splitting.py](#5-stage-4--splittingpy)
6. [Stage 5 — modeling.py](#6-stage-5--modelingpy)
7. [Stage 6 — report.py](#7-stage-6--reportpy)
8. [מודלים וספריות](#8-מודלים-וספריות)
9. [זרימת קבצים](#9-זרימת-קבצים)
10. [שאלות נפוצות בהגנה](#10-שאלות-נפוצות-בהגנה)
11. [הגבלות וסיכונים](#11-הגבלות-וסיכונים)

---

## 1. תמונת על — Pipeline

### מה הפרויקט עושה?

המטרה היא לבנות מערכת שמזהה **סימפטומי PTSD** בהודעות טקסט בעברית — בסלנג צבאי. הנתונים סינתטיים
(נוצרים על ידי LLM, מעוגנים ב-DSM-5), ועוברים EDA, ולידציה אמפירית, חלוקה סטרטיפיקטיבית, ואימון
מודלים, לפני שמופקת השוואה ודוח מסכם.

> **שינוי לעומת גרסאות קודמות:** שלב "Quality Judge" (LLM-as-a-Judge שסינן דגימות לא טבעיות לפני
> ה-EDA) **הוסר מהפייפליין**. `src/data_generation.py` כותב כעת ישירות ל-`dataset.json`, וזה
> מוזרם ישירות ל-EDA (Stage 3) ללא שלב סינון נפרד. אין יותר `quality_judge.py`,
> `judge_cache.json`, `judge_feedback.json`, `quality_metrics.json` או `dataset.clean.json`
> בקוד — קבצים אלה היו שרידים מהגרסה הישנה והוסרו.

```
[data/dataset.json]
      |
      v
[src/eda.py]  <-- ניתוח סטטיסטי + גרפים + בייסליין TF-IDF+LR (Stage 3)
      |
      v
[visuals/01–08,10_*.png]  +  [artifacts/eda_tables.json]  +  [artifacts/baseline_eval.json]
      |
      v
[src/validation.py]  <-- מבחן ריאליזם + ייצוא לתיוג (Stage 3.5, אוטומטי אחרי Stage 3)
      |
      v
[artifacts/realism_test_results.json]  +  [data/annotation_export.csv]
      |
      v
[src/splitting.py]  <-- חלוקה סטרטגית ל-train/test (Stage 4)
      |
      v
[data/train_dataset.json]  +  [data/test_dataset.json]  +  [artifacts/split_manifest.json]
      |
      v
[src/modeling.py]  <-- GPT-4o-mini zero-shot + Fine-tuned SBERT + Fine-tuned AlephBERT (Stage 5)
      |
      v
[artifacts/eval_results.json]  +  [artifacts/misclassified_errors.xlsx]  +  [visuals/09,11_*.png]
      |
      v
[src/report.py]  <-- יצירת מסמכי הדוח (Stage 6)
      |
      v
[reports/slide3_summary.md]  +  [reports/README_eda.md]
```

### הסדר בפועל להרצה:

```bash
python run_pipeline.py                  # כל השלבים, ברצף
python run_pipeline.py --stages 3 4 5    # שלבים בודדים
python run_pipeline.py --skip-validation # בלי Stage 3.5 (אין golden dataset)
```

המספור (1, 3, 3.5, 4, 5, 6) שומר על המספור ההיסטורי של הפייפליין; "2" היה שלב ה-Judge שהוסר, ולכן
המספור מדלג ממנו במכוון.

> **מה צריך לדעת לשאלות:** כל שלב קורא קלט מהדיסק וכותב פלט לדיסק, כך שכל שלב יכול לרוץ בנפרד
> בלי לחזור על השלבים הקודמים — ראו `--stages` ב-`run_pipeline.py`.

---

## 2. קובץ הנתונים — dataset.json

### מבנה הקובץ

`data/dataset.json` הוא מערך (array) של רשומות JSON. כל רשומה מייצגת הודעה סינתטית אחת:

```json
{
  "id": "syn_0000",
  "text": "מאז שהיינו שם בפנים אני מתעצבן מכל שטות...",
  "labels": ["anger_irritability", "functional_impairment", "guilt_shame"],
  "example_type": "positive_clear",
  "platform": "diary",
  "explicitness": "behavioral",
  "severity": "strong",
  "slang_used": [],
  "dsm5_criterion": "E1+E2: Irritable behavior and angry outbursts",
  "synthetic": true
}
```

### תיאור כל שדה

| שדה | סוג | ערכים אפשריים | הסבר |
|-----|-----|---------------|------|
| `id` | string | `"syn_0000"` ... | מזהה ייחודי לדגימה |
| `text` | string | טקסט עברי חופשי | ההודעה הסינתטית עצמה |
| `labels` | list[string] | ראה טבלת labels | תגיות סימפטום — יכולות להיות ריקות (`[]`) |
| `example_type` | string | `positive_clear`, `implicit`, `hard_negative`, `ambiguous` | מידת הבהירות של הסימפטום |
| `platform` | string | `whatsapp`, `tweet`, `reddit`, `diary` | הפלטפורמה שמדמה את ההודעה |
| `explicitness` | string | `explicit`, `implicit`, `behavioral` | איך הסימפטום מוצג — ישיר/רמוז/התנהגותי |
| `severity` | string | `mild`, `medium`, `strong` | עוצמת הסימפטום |
| `slang_used` | list[string] | מילים בסלנג צבאי | המילים הסלנגיות שהופיעו בטקסט |
| `dsm5_criterion` | string | למשל `"E1+E2: ..."` | תת-קריטריון DSM-5 שהנחה את הניסוח (ריק עבור hard-negatives) |
| `synthetic` | bool | `true` | כל הנתונים סינתטיים |

`dsm5_criterion` ושדה `DSM5_CRITERIA` (ב-`src/data_generation.py`) מעגנים את הניסוח קלינית: סמני שפה
ועברית-סלנגית לכל label חולצו ידנית ממסמך קליני וקודדו כקשיחים בקוד — לא מתורגמים בזמן ריצה על ידי
ה-LLM.

### 8 תגיות הסימפטום (Labels)

מספרי התדירות בטבלה הם מ-`artifacts/eda_tables.json` (הרצה אחרונה, N=2000 רשומות):

| תגית (Label) | משמעות | תדירות בדאטאסט |
|-------------|---------|----------------|
| `sleep_disturbance` | הפרעות שינה — אינסומניה, סיוטים, ערות ממושכת | 438 (הנפוץ ביותר) |
| `hypervigilance` | דריכות יתר — סריקת סביבה, בהלה מרעשים | 403 |
| `avoidance` | הימנעות — מניעת מקומות, מצבים, מדיה | 288 |
| `anger_irritability` | כעס ועצבנות — פרצי כעס, סף נמוך לתסכול | 277 |
| `guilt_shame` | אשמה ובושה — האשמה עצמית, survivor guilt | 171 |
| `intrusive_memories` | זיכרונות פולשניים — פלשבקים, מחשבות בלתי רצוניות | 167 |
| `emotional_numbing` | קהות רגשית — ניתוק, חוסר תגובה רגשית | 159 |
| `functional_impairment` | פגיעה בתפקוד — נטישת תחביבים, קשיי עבודה | 133 (הנדיר ביותר) |

### Hard-Negatives — `labels=[]`

500 מתוך 2000 הדגימות הן **hard-negatives** (`example_type = "hard_negative"`). הן מכילות
`"labels": []` — הודעות שנראות דומות לסימפטום אבל אינן כזה (למשל: "עייף מהעבודה" ללא הקשר
טראומטי). הן חיוניות לאימון מסווג שיכול **להבחין** בין סימפטום לבין הודעה רגילה.

> **מה צריך לדעת לשאלות:** hard-negatives לא מוסרות, אלא מטופלות כ"מחלקה תשיעית" (`__NEG__`) בשלב
> ה-split (Stage 4) כדי שהסטרטיפיקציה תוכל לאזן גם אותן.

---

## 3. Stage 3 — eda.py (EDA + בייסליין TF-IDF/LR)

### מטרה

ניתוח סטטיסטי מלא של הנתונים הגולמיים (אין יותר שלב סינון שקודם לכך), כולל יצירת 8 גרפי EDA
ב-`visuals/`, ולאחר מכן — אם קיימים `train_dataset.json`/`test_dataset.json` מ-Stage 4 — אימון
בייסליין TF-IDF + Logistic Regression והפקת עוד שני גרפים (09, 10).

---

### `load_dataset` — טעינה ובינאריזציה

```python
def load_dataset(path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    with open(path, encoding="utf-8-sig") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    df["word_count"] = df["text"].apply(lambda t: len(str(t).split()))
    df["char_count"] = df["text"].apply(lambda t: len(str(t)))
    df["label_count"] = df["labels"].apply(len)
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["labels"])
    classes = list(mlb.classes_)
    return df, Y, classes
```

**`MultiLabelBinarizer`** (מ-`sklearn.preprocessing`):
- הופך רשימת labels לוקטור בינארי.
- `Y` היא מטריצה בצורת `(2000, 8)` — שורה לכל רשומה, עמודה לכל label.
- `classes` = רשימה מסודרת של שמות ה-labels.

**עמודות חישוביות שנוספות:** `word_count`, `char_count`, `label_count` (0 עבור hard-negatives).

---

### RTL Pipeline — רינדור עברית ב-matplotlib

```python
import arabic_reshaper
from bidi.algorithm import get_display as bidi_display

def rtl(text: str) -> str:
    return bidi_display(arabic_reshaper.reshape(text))
```

**הבעיה:** matplotlib מרנדר טקסט משמאל-לימין (LTR). עברית היא ימין-לשמאל (RTL). בלי טיפול, אותיות
עבריות יופיעו הפוכות או לא מסודרות.

**הפתרון:** `arabic_reshaper.reshape` מחבר אותיות לצורות מבודדות, ואז `bidi_display` מחיל את
אלגוריתם Unicode BiDi. מוחל **רק על תוויות צירים ומקרא** — לא על הנתונים עצמם. אם החבילות לא
מותקנות, `eda.py` מציג אזהרה וממשיך עם טקסט גולמי (פונקציית `rtl` הופכת ל-no-op).

---

### 8 הגרפים העיקריים — פירוט

| קובץ | מה מוצג | חישוב |
|------|---------|-------|
| `01_label_marginals.png` | תדירות שולית לכל label | `Y.sum(axis=0)` |
| `02_label_cooccurrence.png` | מטריצת שיתוף-הופעה 8×8 | `Y.T @ Y` |
| `02b_label_correlation.png` | מתאם פירסון (phi coefficient) בין labels | `np.corrcoef(Y.T)` |
| `03_cardinality.png` | התפלגות מספר labels לרשומה (0=hard-negative) | `Counter(df["label_count"])` |
| `04_length_by_platform.png` | אורך טקסט (מילים/תווים) לפי פלטפורמה | violin plot |
| `05_platform_x_type.png` | פלטפורמה × סוג דגימה | `pd.crosstab` |
| `06_severity_x_label.png` | חומרה × label | heatmap |
| `07_explicitness_x_label.png` | מפורשות × label | heatmap |
| `08_slang_top20.png` | 20 מילות הסלנג הנפוצות | `Counter(all_slang).most_common(20)` |

מעבר ל-8 הגרפים האלה, `eda.py` מפיק שלושה גרפים נוספים שתלויים בשלבים מאוחרים יותר:

| קובץ | מקור | תנאי |
|------|------|------|
| `09_model_comparison.png` | F1-micro של כל המודלים (TF-IDF, GPT-4o-mini, SBERT, AlephBERT) | רץ אם `eval_results.json` ו/או `baseline_eval.json` קיימים |
| `10_tfidf_splits.png` | F1 לכל label — TF-IDF train מול test | רץ כחלק מ-`run_tfidf_lr_baseline()` |
| `11_perlabel_finetuned.png` | Precision/Recall/F1 לכל label — SBERT המאומן | רץ אם `eval_results.json` קיים (אחרי Stage 5) |

---

### `run_tfidf_lr_baseline` — בייסליין TF-IDF + Logistic Regression

מאומן ומוערך **כאן**, בתוך `eda.py` — לא ב-`modeling.py`. משתמש ב-`TfidfVectorizer`
(unigrams+bigrams, `sublinear_tf=True`) ו-`OneVsRestClassifier(LogisticRegression(class_weight="balanced"))`.
מבצע threshold-tuning per-label על carve-out מתוך ה-train, ושומר תוצאות ל-`artifacts/baseline_eval.json`.

תוצאה בפועל (הרצה אחרונה): **F1-micro=0.849 / F1-macro=0.821 / Accuracy=0.74** על סט הבדיקה.

> **מה צריך לדעת לשאלות:** הסיבה שהבייסליין הפשוט יושב ב-EDA ולא ב-modeling היא הפרדת אחריות —
> מודלים פשוטים/קלאסיים משויכים לניתוח החוקר, ושלב המודלים (Stage 5) מתמקד במודלים נוירונים
> שמתאמנים בפועל.

---

### `build_eda_tables` — טבלאות מספריות

מייצר `artifacts/eda_tables.json` עם `n_records`, `label_marginals`, `platform_distribution`,
`example_type_distribution`, `severity_distribution`, `text_length_stats`,
`label_cardinality_distribution`, `mean_label_cardinality`, `slang_coverage_rate`.

---

## 4. Stage 3.5 — validation.py

> **שלב חדש שלא היה בגרסה הקודמת** — מחליף את שלב ה-Judge שהוסר. במקום שה-LLM יסנן דגימות בזמן
> הריצה, השלב הזה מודד **אמפירית** עד כמה הדאטה הסינתטי קרוב לטקסט אנושי אמיתי, ומכין דגימה
> לתיוג ידני. רץ אוטומטית אחרי Stage 3 (אלא אם `--skip-validation`).

### מבחן ריאליזם — `run_realism_test`

מזווג כל משפט מ-`data/golden_dataset.json` (טקסט אנושי אמיתי, ראו `scrape_reddit.py`) עם משפט
סינתטי אקראי, ומבקש מה-LLM לזהות מי מהשניים נכתב על ידי מכונה. שיעור זיהוי **≥ 85%** מדגיל אזהרה —
סימן שהטקסט הסינתטי "נראה מלאכותי" מדי. תוצאה נכתבת ל-`artifacts/realism_test_results.json`.

הקובץ `data/golden_dataset.json` **לא מיוצר על ידי הפייפליין** — חייבים לספק אותו (למשל באמצעות
`scrape_reddit.py`, שאוסף תגובות מ-r/ptsd ושומר אותן בפורמט המתאים). אם הוא חסר, Stage 3.5 מדלג
על מבחן הריאליזם אך עדיין מריץ את ייצוא התיוג.

### ייצוא לתיוג — `export_for_annotation`

דוגם 100 משפטים אקראיים מ-`data/dataset.json` ושומר ל-`data/annotation_export.csv` עם עמודות
`auto_label_{label}` (תיוג אוטומטי, לרפרנס) ו-`annotation_{label}` (ריקות — למתייגים למלא).

### Cohen's Kappa — `compute_cohen_kappa`

**שלב ידני, לא רץ אוטומטית** מתוך `run_pipeline.py`. כאשר שני מתייגים ממלאים בנפרד עותקים של
`annotation_export.csv` (נשמרים בדרך כלל בשמות `data/annotation_rater1.csv` /
`annotation_rater2.csv`), מריצים את הפונקציה הזו ידנית כדי לחשב Cohen's Kappa לכל label, ולשמור
ל-`artifacts/kappa_results.json`.

---

## 5. Stage 4 — splitting.py

### מטרה

לחלק את הנתונים ל-**75% אימון** ו-**25% בדיקה** בצורה שמבטיחה ייצוג של **כל label** בשני הסטים.

### למה `sklearn.train_test_split` הרגיל לא עובד?

`stratify=` מקבל וקטור חד-ממדי. עבור multi-label, נצטרך להפוך את ה-labels ל"class powerset" (כל
**שילוב** של labels = class נפרד) — עם מאות שילובים ייחודיים, רבים מהם מופיעים פעם אחת בלבד.
sklearn ידחה אותם עם `ValueError`.

### הפתרון: Iterative Stratification + `_NEG_LABEL`

`run_split_pipeline` מזריק label סינתטי `"__NEG__"` לרשומות hard-negative (labels ריקות) **רק
לצורך הסטרטיפיקציה**, מריץ `iterative_train_test_split` (Sechidis et al., 2011, via
`scikit-multilearn`), ואז מסיר את `__NEG__` מהפלט.

> **חשוב — מצב נוכחי בפועל:** `artifacts/split_manifest.json` הנוכחי מציין
> `"stratifier": "random_fallback"`, כלומר בריצה האחרונה `scikit-multilearn` לא היה מותקן
> בסביבה, וההרצה חזרה לפול-בק אקראי (`_random_stratified_split`) במקום ל-Iterative
> Stratification. הפייפליין עדיין עובד וה-sanity gates עדיין עוברים, אבל מבחינת דיוק תיאורי —
> אם רוצים את שיטת Sechidis בפועל, יש להתקין `scikit-multilearn` (`pip install scikit-multilearn`)
> ולהריץ Stage 4 מחדש לפני ההגשה.

### Sanity Gates — שלושה שערי בטיחות

| Gate | תנאי |
|------|------|
| 1 | כל label מופיע לפחות פעם אחת ב-test |
| 2 | לכל label, `train_share ∈ [0.65, 0.85]` |
| 3 | JS divergence בין train ל-test < 0.05 |

תוצאה בפועל (הרצה אחרונה, N=2000): **JS divergence = 0.0023** — מתחת לסף בהרבה.

### קבצי פלט של Stage 4

| קובץ | תוכן (הרצה אחרונה) |
|------|------|
| `data/train_dataset.json` | 1500 רשומות (75%) |
| `data/test_dataset.json` | 500 רשומות (25%) |
| `artifacts/split_manifest.json` | seed, ratios, JS divergence, סוג הסטרטיפיקטור, גרסת scikit-multilearn |

---

## 6. Stage 5 — modeling.py

### מטרה

להעריך שלושה מודלים מתקדמים על אותו סט בדיקה ולשמור את התוצאות ל-`artifacts/eval_results.json`:

| מודל | סוג | תיאור |
|------|-----|-------|
| GPT-4o-mini | Zero-shot | OpenAI structured outputs — בלי אימון, דורש `OPENAI_API_KEY` |
| Fine-tuned SBERT | מאומן | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, mean-pool + linear head, 10 epochs |
| Fine-tuned AlephBERT | מאומן | `dicta-il/alephbertgimmel-base` — Hebrew-native, ארכיטקטורה זהה ל-SBERT |

תוצאות בפועל (הרצה אחרונה, F1-micro על סט הבדיקה):

| מודל | F1-micro | Accuracy |
|------|----------|----------|
| GPT-4o-mini (zero-shot) | 0.673 | 0.384 |
| TF-IDF + LR (Stage 3) | 0.849 | 0.74 |
| Fine-tuned SBERT | 0.871 | 0.758 |
| Fine-tuned AlephBERT | **0.933** | **0.87** |

AlephBERT (מודל עברי ייעודי) עוקף את SBERT הרב-לשוני בכל המדדים — תוצאה סבירה, מודל שאומן
ספציפית על עברית מבין רישום וסלנג עברי טוב יותר ממודל רב-לשוני כללי.

לאחר שני המודלים המאומנים, `run_modeling_pipeline` מייצא טעויות סיווג (FP/FN) של SBERT ל-
`artifacts/misclassified_errors.xlsx` לניתוח ידני, ומפעיל מחדש את `plot_model_comparison()` /
`plot_perlabel_finetuned()` מ-`eda.py` כדי לעדכן את גרפים 09 ו-11.

---

## 7. Stage 6 — report.py

### מטרה

לאסוף את תוצאות Stages 3–5 ולייצר מסמכי דוח סופיים: `reports/slide3_summary.md` ו-
`reports/README_eda.md`.

`render_slide3` טוען את `split_manifest.json` ו-`eda_tables.json`, מחשב N, מספר labels, cardinality
ממוצע, label נדיר/נפוץ, JS divergence, ו-N train/test, ומזין אותם לתבנית `_SLIDE3_TEMPLATE`.

`render_readme` יוצר `README_eda.md` עם טבלת markdown שמקשרת כל קובץ PNG (8 גרפי EDA הליבה,
לפי `_VISUAL_CAPTIONS`) לתיאור.

`verify_charts` בודק שכל הגרפים קיימים וגדולים מ-20KB (גרף קטן מדי מרמז על שגיאת שמירה).

> **מה צריך לדעת לשאלות:** `run_report_pipeline(run_eda=False)` (כך `run_pipeline.py` קורא לו
> ב-Stage 6) מדלג על הרצת EDA מחדש כיוון ש-Stage 3 כבר רץ קודם באותה הרצה.

---

## 8. מודלים וספריות

### LLM Providers (Stage 1 — generation, Stage 3.5 — validation)

`src/data_generation.py` תומך בכמה ספקי LLM להחלפה דרך `LLMProvider`:

| ספק | שימוש בפועל |
|-----|-------------|
| OpenAI (`gpt-4o-mini`) | ספק ראשי כש-`OPENAI_API_KEY` מוגדר — לדור הדאטה ולמבחן הריאליזם |
| Ollama (מקומי, למשל `llama3`) | זוהה אוטומטית אם רץ מקומית ואין מפתח API בענן |
| OpenRouter (free tier) | פולבק חינמי אם אין OpenAI/Ollama |
| Gemini | קיים בקוד (`GeminiClient`) כפולבק נוסף ב-`ResilienceLLMClient`, ודרך `data_generation.py` כקובץ עצמאי, אך לא נבחר על ידי `run_pipeline.py` |
| MockLLMClient | פולבק דטרמיניסטי (טקסט עברי קבוע) כש-`--mock` או כשאין שום ספק זמין |

### ספריות Python (לפי שלב)

| ספרייה | שימוש |
|--------|-------|
| `scikit-multilearn` | `iterative_train_test_split` (Stage 4) |
| `scikit-learn` | `MultiLabelBinarizer`, `TfidfVectorizer`, `LogisticRegression`, metrics |
| `pandas` | DataFrame, crosstab, ייצוא Excel |
| `numpy` | מטריצות labels, JS divergence |
| `matplotlib` / `seaborn` | כל הגרפים |
| `arabic-reshaper` / `python-bidi` | רינדור RTL לעברית בגרפים |
| `transformers` / `torch` | Fine-tuning ל-SBERT ו-AlephBERT (Stage 5) |
| `openai` | GPT-4o-mini zero-shot baseline ומבחן ריאליזם |
| `openpyxl` | ייצוא `misclassified_errors.xlsx` |

---

## 9. זרימת קבצים

```
INPUT
└── data/dataset.json              (פלט Stage 1 — נתוני גלם)

STAGE 3 (eda.py)
├── IN:  data/dataset.json
├── OUT: visuals/01_label_marginals.png ... 08_slang_top20.png
├── OUT: artifacts/eda_tables.json
└── (אם train/test קיימים) OUT: artifacts/baseline_eval.json, visuals/09,10_*.png

STAGE 3.5 (validation.py)
├── IN:  data/dataset.json, data/golden_dataset.json (לא חובה)
├── OUT: data/annotation_export.csv
└── OUT: artifacts/realism_test_results.json (אם golden dataset קיים)

STAGE 4 (splitting.py)
├── IN:  data/dataset.json
├── OUT: data/train_dataset.json   (1500 רשומות, ~75%)
├── OUT: data/test_dataset.json    (500 רשומות, ~25%)
└── OUT: artifacts/split_manifest.json

STAGE 5 (modeling.py)
├── IN:  data/train_dataset.json, data/test_dataset.json
├── OUT: artifacts/eval_results.json
├── OUT: artifacts/misclassified_errors.xlsx
└── OUT: visuals/09_model_comparison.png, visuals/11_perlabel_finetuned.png

STAGE 6 (report.py)
├── IN:  artifacts/eda_tables.json, artifacts/split_manifest.json
├── IN:  visuals/*.png  (verify_charts בודק קיום + גודל)
├── OUT: reports/slide3_summary.md
└── OUT: reports/README_eda.md
```

---

## 10. שאלות נפוצות בהגנה

### 1. למה Iterative Stratification ולא `stratify=` הרגיל של sklearn?

**תשובה:** `sklearn.train_test_split(stratify=y)` מקבל וקטור חד-ממדי. עבור multi-label היינו
צריכים ליצור "powerset classes" — כל שילוב ייחודי של labels הופך ל-class, ועם מאות שילובים רבים
מהם בודדים, sklearn ידחה עם `ValueError`. Iterative Stratification (Sechidis et al., 2011) מטפל
ב-multi-label natively, ומבטיח שכל label יהיה מיוצג בשני הסטים.

---

### 2. למה 75/25 ולא 80/20?

**תשובה:** עם N=2000 ו-8 labels, ה-test set (500 רשומות) עדיין נותן ייצוג מספיק אפילו ל-label
הנדיר ביותר (`functional_impairment`, 133 דגימות כולל — 37 ב-test, ראו `split_manifest.json`).
75/25 משאיר מספיק נתוני אימון בלי לרושש את ה-test מ-labels נדירים.

---

### 3. מה זה JS Divergence ולמה הסף הוא 0.05?

**תשובה:** Jensen-Shannon Divergence מודד דמיון בין שתי התפלגויות הסתברות (label marginals של
train ו-test). הסף 0.05 שרירותי אך מקובל בספרות — מתחתיו, ההתפלגויות "דומות מספיק" שמסווג
שמתאמן על train יכליל היטב על test. בפרויקט הנוכחי JS=0.0023 — הרבה מתחת לסף.

---

### 4. למה hard-negatives חשובות?

**תשובה:** מסווג שרואה רק דגימות חיוביות ילמד "הכל הוא סימפטום". hard-negatives (500 מ-2000
דגימות) מלמדות אותו לדחות הודעות שנראות דומות אבל אינן טראומטיות. בסלנג צבאי זה קריטי — "עייף
מהאימון" ≠ `sleep_disturbance`.

---

### 5. למה הוחלף שלב ה-Quality Judge במבחן ריאליזם (Stage 3.5)?

**תשובה:** שלב ה-Judge הישן השתמש ב-LLM כדי לסנן דגימות *לפני* האימון, על בסיס שיפוט סובייקטיבי
("האם זה נשמע טבעי?"). מבחן הריאליזם (Stage 3.5) הוא מדד **אמפירי**: הוא מודד כמה שניתן להבחין
בין הסינתטי לאמיתי, על ידי השוואה ישירה למשפטים אנושיים אמיתיים (`golden_dataset.json`), ומפיק
מספר (detection rate) שניתן להשוות בין הרצות — בלי לסנן/לזרוק דגימות בדרך.

---

### 6. למה ה-baseline TF-IDF+LR נמצא ב-`eda.py` ולא ב-`modeling.py`?

**תשובה:** הפרדת אחריות — `eda.py` מטפל בניתוח ובמודלים קלאסיים/פשוטים שמשמשים כנקודת השוואה
(reference point), ו-`modeling.py` מתמקד במודלים הנוירונים שבפועל מאומנים על GPU (SBERT,
AlephBERT) ובמודל ה-API (GPT-4o-mini).

---

### 7. מה זה `MultiLabelBinarizer` ולמה צריך אותו?

**תשובה:** `sklearn.preprocessing.MultiLabelBinarizer` הופך רשימת lists של labels למטריצה
בינארית. זה נחוץ כי אלגוריתמי ML עובדים עם מספרים, `iterative_train_test_split` מצפה למטריצה Y
בצורת (n_samples, n_classes), וחישובי co-occurrence (`Y.T @ Y`) אפשריים רק על מטריצה נומרית.

---

### 8. למה הנתונים סינתטיים ולא אמיתיים?

**תשובה:** נתונים קליניים אמיתיים על PTSD מצריכים אישורי IRB (ועדת אתיקה), הגנת פרטיות, ואין
מאגר ציבורי בעברית עם אנוטציה multi-label. `data_generation.py` יוצר נתונים מבוקרים — ניתן לשלוט
ב-platform, severity, explicitness, slang level, ומעוגן ב-DSM-5. החיסרון: הנתונים לא מייצגים את
ה-variance האמיתי של דיבור אנשים שחוו טראומה — זו הסיבה שנוסף מבחן הריאליזם ב-Stage 3.5.

---

### 9. למה AlephBERT הציג תוצאות טובות יותר מ-SBERT הרב-לשוני?

**תשובה:** AlephBERT (`dicta-il/alephbertgimmel-base`) אומן ספציפית על קורפוס עברי גדול, ומבין
טוב יותר רישום שפה, סלנג, ותחביר עברי. SBERT הרב-לשוני (`paraphrase-multilingual-MiniLM-L12-v2`)
חולק את הקיבולת שלו בין עשרות שפות, ולכן מבין עברית "ברמה כללית" אבל מפסיד דיוק בניואנסים
ספציפיים לשפה — כפי שניכר ב-F1-macro (0.916 לעומת 0.836).

---

## 11. הגבלות וסיכונים

### א. סטרטיפיקציה — תלות בהתקנה

ה-split (Stage 4) משתמש ב-Iterative Stratification רק אם `scikit-multilearn` מותקן; אחרת חוזר
לפול-בק אקראי עם אזהרה בלוג. ה-manifest הנוכחי (`split_manifest.json`) מציין
`"stratifier": "random_fallback"` — כלומר ההרצה האחרונה לא השתמשה בשיטת Sechidis בפועל. אם
מציגים את הפרויקט כמשתמש ב-Iterative Stratification, יש לוודא ש-`scikit-multilearn` מותקן ולהריץ
Stage 4 מחדש לפני ההגשה.

### ב. מקור סינתטי בלבד

כל הנתונים נוצרו על ידי LLM. גם עם עיגון DSM-5 ובדיקת ריאליזם, יש סיכון ל-variance מוגבל — דגימות
מאותו label/platform עשויות "להרגיש דומות" כי נוצרו על ידי אותו מודל עם אותו פרומפט-פקטורי.

### ג. מבחן הריאליזם תלוי ב-golden dataset קטן ובשופט LLM יחיד

`run_realism_test` משווה למשפטים אמיתיים מתוך `golden_dataset.json` (שנאסף ידנית, היקף מצומצם),
ומשתמש ב-LLM יחיד כ"שופט" אם הטקסט סינתטי או אנושי. שופט יחיד עלול לפספס הבדלים עדינים, ו-N קטן
של golden dataset מגביל את מובהקות המבחן.

### ד. אין Inter-Annotator Agreement מאומת על כל הדאטה

`compute_cohen_kappa` קיים ומחשב הסכמה בין שני מתייגים, אך זהו שלב ידני — אם לא הורץ בפועל על
`data/annotation_rater1.csv` / `annotation_rater2.csv`, אין עדיין מדד IAA מאומת על הדאטה הנוכחי.
יש לוודא ש-`artifacts/kappa_results.json` קיים ומשקף תיוג בפועל לפני שמציגים אותו כממצא.

### ה. GPT-4o-mini zero-shot חלש ביחס למודלים המאומנים

F1-micro=0.673 ו-Accuracy=0.384 בלבד עבור GPT-4o-mini zero-shot, לעומת 0.87–0.93 למודלים
המאומנים. זה תקין ומצופה (zero-shot מול fine-tuned), אבל חשוב להציג זאת כהשוואת בייסליין ולא
כממצא שלילי על GPT-4o-mini עצמו — המודל לא אומן על הדומיין הספציפי הזה.

---

*מסמך זה מתאר את ה-Pipeline כפי שמומש בקובצי המקור הנוכחיים (`src/`), ללא שלב ה-Quality Judge
שהוסר. המספרים בטבלאות מבוססים על תוכן `artifacts/` בפועל בעת כתיבת המסמך — אם מריצים את
הפייפליין מחדש עם נתונים/seed שונים, יש לעדכן אותם.*
