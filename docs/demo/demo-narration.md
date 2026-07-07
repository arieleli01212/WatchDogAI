# WatchDogAI — Demo Video Narration Script

Narration script for `docs/demo/WatchDogAI_demo.mp4` (1:55). Each section lists the
on-screen visual, the timestamp, and the spoken Hebrew text — use this to
re-record the voiceover (your own voice / ElevenLabs etc.) over the silent
video parts.

| # | Time | Visual | Audio file |
|---|------|--------|-----------|
| 1 | 0:00–0:16 | Title slide | `n1` |
| 2 | 0:16–0:45 | Architecture slide ("How it works") | `n2` |
| 3 | 0:45–0:52 | "Demo 1 — Normal footage" slide | `n3` |
| 4 | 0:52–1:02 | Restaurant CCTV + live model overlay | `n3b` |
| 5 | 1:02–1:08 | "Demo 2 — Violent footage" slide | `n4` |
| 6 | 1:08–1:25 | Fight clip + overlay, alert fires | `n4b` |
| 7 | 1:25–1:46 | Dashboard: live view, then alerts page | `n5` |
| 8 | 1:46–1:55 | Outro slide | `n6` |

---

## 1. פתיח (שקופית כותרת)

שלום, אנחנו שמחים להציג את WatchDogAI — מערכת חכמה לזיהוי אלימות בזמן אמת.
המערכת מקבלת וידאו ממצלמת אבטחה או מקובץ, מזהה אירועי אלימות בעזרת מודל
בינה מלאכותית, מקליטה אוטומטית קליפ של האירוע, ומציגה התרעות בדשבורד חי.

## 2. איך זה עובד (שקופית ארכיטקטורה)

כך המערכת עובדת. תהליך הצילום קורא פריימים מהמצלמה בזמן אמת. כל פריים נשלח
למודל Vision Transformer שאומן לזהות אלימות, והמודל מחזיר סיווג — נורמלי או
אלימות — יחד עם רמת ביטחון. כדי למנוע התרעות שווא, המערכת דורשת שלושה
זיהויים רצופים מעל סף ביטחון של 85 אחוזים. רק כשאלימות מאושרת, מוקלט קליפ
MP4 שכולל גם את השניות שלפני האירוע, וההתרעה נשמרת במסד הנתונים.

## 3. דמו 1 — פתיח

בדוגמה הראשונה, המערכת מקבלת סרטון ממצלמת אבטחה במסעדה — סיטואציה רגילה
לחלוטין.

## 4. דמו 1 — על גבי הסרטון

אפשר לראות למעלה שהמודל מסווג כל פריים כנורמלי. רמת הביטחון לאלימות עומדת
על כשמונה אחוזים בלבד, הרחק מתחת לסף, ולכן לא נוצרת שום התרעה.

## 5. דמו 2 — פתיח

בדוגמה השנייה, המערכת מקבלת סרטון של קטטה שצולמה במצלמת אבטחה.

## 6. דמו 2 — על גבי הסרטון

כאן רמת הביטחון קופצת מעל תשעים אחוזים. אחרי שלושה זיהויים רצופים מעל הסף,
המערכת מכריזה על אלימות מאומתת. ברגע הזה מתחילה הקלטה של קליפ האירוע, כולל
השניות שקדמו לו, וההתרעה נשמרת במסד הנתונים עם חותמת זמן ורמת ביטחון.

## 7. דשבורד

המערכת כוללת גם דשבורד אינטרנטי חי. במסך הראשי רואים את המצלמה בשידור חי,
כולל זיהוי ומעקב אחרי אנשים בעזרת YOLO, ואת סטטוס האלימות הנוכחי. כשמצלמה
נמצאת במצב התרעה, היא מסומנת באדום. במסך ההתרעות מופיעה ההיסטוריה המלאה של
האירועים, עם תאריך ושעה, רמת ביטחון, והקליפ ששמור לכל אירוע.

## 8. סיום

זו הייתה הצגה של WatchDogAI. המודל קיבל שני סרטונים, סיווג נכון את שניהם,
והתריע רק כשבאמת היה צריך. תודה על הצפייה.

---

**Demo footage source:** HuggingFace dataset
`valiantlynxz/godseye-violence-detection-dataset` (Fight / NonFight
surveillance clips). The detection overlays were rendered by running the
project's own `ViolenceDetector` on every frame; the dashboard shots are
screenshots of the real app running with `CAMERA_SOURCE` set to the fight
clip.
