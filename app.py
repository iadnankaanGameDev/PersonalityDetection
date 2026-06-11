import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

with open("personality_model_package.pkl", "rb") as f:
    model_package = joblib.load("personality_model_package.pkl")

model = model_package["model"]
features = model_package["features"]

TRANSLATIONS = {
    "en": {
        "page_badge": "Multi-class Classification · Random Forest · Synthetic Dataset",
        "title": "Personality Classification",
        "subtitle": "This educational machine learning app predicts one of 16 personality classes based on questionnaire-style responses. Each question is answered on a scale from strongly disagree to strongly agree.",
        "warning": "Important: This model was trained on a synthetic dataset. It should not be used as a real psychological assessment tool.",
        "questionnaire_title": "Questionnaire Inputs",
        "questionnaire_subtitle": "Select a value for each question. The scale ranges from -3 to 3.",
        "predict_button": "Predict Personality",
        "reset_button": "Reset Answers",
        "predicted_personality": "Predicted Personality",
        "top_3_predictions": "Top 3 Predictions",
        "confidence_label": "Model Vote Confidence",
        "confidence_explanation": "Based on Random Forest vote distribution, not a psychological certainty.",
        "low_confidence": "Low confidence: your answers may match multiple personality patterns.",
        "moderate_confidence": "Moderate confidence: there are some close alternative predictions.",
        "high_confidence": "High confidence: the model found a clear dominant pattern.",
    },
    "tr": {
        "page_badge": "Çok Sınıflı Sınıflandırma · Random Forest · Sentetik Veri Seti",
        "title": "Kişilik Sınıflandırma",
        "subtitle": "Bu eğitim amaçlı makine öğrenmesi uygulaması, anket tarzı cevaplara göre 16 kişilik sınıfından birini tahmin eder. Her soru, kesinlikle katılmıyorum ile kesinlikle katılıyorum arasında puanlanır.",
        "warning": "Önemli: Bu model sentetik bir veri setiyle eğitilmiştir. Gerçek bir psikolojik değerlendirme aracı olarak kullanılmamalıdır.",
        "questionnaire_title": "Anket Cevapları",
        "questionnaire_subtitle": "Her soru için bir değer seçin. Ölçek -3 ile 3 arasındadır.",
        "predict_button": "Kişilik Tipini Tahmin Et",
        "reset_button": "Cevapları Sıfırla",
        "predicted_personality": "Tahmin Edilen Kişilik",
        "top_3_predictions": "En Olası 3 Tahmin",
        "confidence_label": "Model Oy Güveni",
        "confidence_explanation": "Random Forest ağaçlarının oy dağılımına dayanır; psikolojik kesinlik anlamına gelmez.",
        "low_confidence": "Düşük güven: cevaplarınız birden fazla kişilik örüntüsüyle benzerlik gösterebilir.",
        "moderate_confidence": "Orta güven: birbirine yakın alternatif tahminler bulunuyor.",
        "high_confidence": "Yüksek güven: model belirgin bir baskın örüntü buldu.",
    },
}

SCALE_LABELS = {
    "en": {
        -3: "Strongly Disagree",
        -2: "Disagree",
        -1: "Slightly Disagree",
        0: "Neutral",
        1: "Slightly Agree",
        2: "Agree",
        3: "Strongly Agree",
    },
    "tr": {
        -3: "Kesinlikle Katılmıyorum",
        -2: "Katılmıyorum",
        -1: "Biraz Katılmıyorum",
        0: "Nötr",
        1: "Biraz Katılıyorum",
        2: "Katılıyorum",
        3: "Kesinlikle Katılıyorum",
    },
}

QUESTION_LABELS_TR = {
    "You regularly make new friends.": "Düzenli olarak yeni arkadaşlar edinirsiniz.",
    "You spend a lot of your free time exploring various random topics that pique your interest": "Boş zamanınızın çoğunu ilginizi çeken farklı konuları keşfetmeye ayırırsınız.",
    "Seeing other people cry can easily make you feel like you want to cry too": "Başkalarının ağladığını görmek sizin de kolayca ağlama isteği duymanıza neden olabilir.",
    "You often make a backup plan for a backup plan.": "Çoğu zaman yedek planınız için bile bir yedek plan yaparsınız.",
    "You usually stay calm, even under a lot of pressure": "Yoğun baskı altında bile genellikle sakin kalırsınız.",
    "At social events, you rarely try to introduce yourself to new people and mostly talk to the ones you already know": "Sosyal etkinliklerde yeni insanlarla tanışmaya nadiren çalışır, çoğunlukla zaten tanıdığınız kişilerle konuşursunuz.",
    "You prefer to completely finish one project before starting another.": "Başka bir projeye başlamadan önce bir projeyi tamamen bitirmeyi tercih edersiniz.",
    "You are very sentimental.": "Oldukça duygusalsınızdır.",
    "You like to use organizing tools like schedules and lists.": "Programlar ve listeler gibi düzenleme araçlarını kullanmayı seversiniz.",
    "Even a small mistake can cause you to doubt your overall abilities and knowledge.": "Küçük bir hata bile genel yeteneklerinizden ve bilginizden şüphe etmenize neden olabilir.",
    "You feel comfortable just walking up to someone you find interesting and striking up a conversation.": "İlginizi çeken birinin yanına gidip sohbet başlatmak size rahat gelir.",
    "You are not too interested in discussing various interpretations and analyses of creative works.": "Yaratıcı eserlerin farklı yorumlarını ve analizlerini tartışmakla pek ilgilenmezsiniz.",
    "You are more inclined to follow your head than your heart.": "Kalbinizden çok aklınızı dinlemeye daha yatkınsınızdır.",
    "You usually prefer just doing what you feel like at any given moment instead of planning a particular daily routine.": "Belirli bir günlük rutin planlamak yerine çoğu zaman o an içinizden geleni yapmayı tercih edersiniz.",
    "You rarely worry about whether you make a good impression on people you meet.": "Tanıştığınız insanlar üzerinde iyi bir izlenim bırakıp bırakmadığınız konusunda nadiren endişelenirsiniz.",
    "You enjoy participating in group activities.": "Grup aktivitelerine katılmaktan keyif alırsınız.",
    "You like books and movies that make you come up with your own interpretation of the ending.": "Sonunu kendi yorumunuzla anlamlandırmanızı sağlayan kitapları ve filmleri seversiniz.",
    "Your happiness comes more from helping others accomplish things than your own accomplishments.": "Mutluluğunuz, kendi başarılarınızdan çok başkalarının bir şeyler başarmasına yardım etmekten gelir.",
    "You are interested in so many things that you find it difficult to choose what to try next.": "O kadar çok şeye ilgi duyarsınız ki sırada ne deneyeceğinizi seçmekte zorlanırsınız.",
    "You are prone to worrying that things will take a turn for the worse.": "İşlerin kötüye gidebileceği konusunda endişelenmeye yatkınsınızdır.",
    "You avoid leadership roles in group settings.": "Grup ortamlarında liderlik rollerinden kaçınırsınız.",
    "You are definitely not an artistic type of person.": "Kesinlikle sanatsal yönü güçlü biri değilsinizdir.",
    "You think the world would be a better place if people relied more on rationality and less on their feelings.": "İnsanlar duygularından çok mantığa güvenirse dünyanın daha iyi bir yer olacağını düşünürsünüz.",
    "You prefer to do your chores before allowing yourself to relax.": "Rahatlamadan önce işlerinizi bitirmeyi tercih edersiniz.",
    "You enjoy watching people argue.": "İnsanların tartışmasını izlemekten keyif alırsınız.",
    "You tend to avoid drawing attention to yourself.": "Dikkatleri üzerinize çekmekten kaçınma eğilimindesinizdir.",
    "Your mood can change very quickly.": "Ruh haliniz çok hızlı değişebilir.",
    "You lose patience with people who are not as efficient as you.": "Sizin kadar verimli olmayan insanlara karşı sabrınızı kaybedersiniz.",
    "You often end up doing things at the last possible moment.": "Çoğu zaman işleri son mümkün anda yaparsınız.",
    "You have always been fascinated by the question of what, if anything, happens after death.": "Ölümden sonra ne olup olmadığı sorusu sizi her zaman büyülemiştir.",
    "You usually prefer to be around others rather than on your own.": "Genellikle yalnız olmaktansa başkalarının yanında olmayı tercih edersiniz.",
    "You become bored or lose interest when the discussion gets highly theoretical.": "Tartışma çok teorik hale geldiğinde sıkılır veya ilginizi kaybedersiniz.",
    "You find it easy to empathize with a person whose experiences are very different from yours.": "Deneyimleri sizinkinden çok farklı olan biriyle empati kurmayı kolay bulursunuz.",
    "You usually postpone finalizing decisions for as long as possible.": "Kararları kesinleştirmeyi genellikle mümkün olduğunca ertelersiniz.",
    "You rarely second-guess the choices that you have made.": "Verdiğiniz kararları nadiren tekrar sorgularsınız.",
    "After a long and exhausting week, a lively social event is just what you need.": "Uzun ve yorucu bir haftadan sonra canlı bir sosyal etkinlik tam da ihtiyacınız olan şeydir.",
    "You enjoy going to art museums.": "Sanat müzelerine gitmekten keyif alırsınız.",
    "You often have a hard time understanding other people’s feelings.": "Başkalarının duygularını anlamakta sık sık zorlanırsınız.",
    "You like to have a to-do list for each day.": "Her gün için yapılacaklar listesi olmasını seversiniz.",
    "You rarely feel insecure.": "Kendinizi nadiren güvensiz hissedersiniz.",
    "You avoid making phone calls.": "Telefon görüşmesi yapmaktan kaçınırsınız.",
    "You often spend a lot of time trying to understand views that are very different from your own.": "Sizinkinden çok farklı görüşleri anlamaya çalışmak için sık sık çok zaman harcarsınız.",
    "In your social circle, you are often the one who contacts your friends and initiates activities.": "Sosyal çevrenizde arkadaşlarınızla iletişime geçen ve etkinlikleri başlatan kişi çoğu zaman sizsinizdir.",
    "If your plans are interrupted, your top priority is to get back on track as soon as possible.": "Planlarınız kesintiye uğrarsa en önemli önceliğiniz mümkün olduğunca hızlı şekilde tekrar düzene dönmektir.",
    "You are still bothered by mistakes that you made a long time ago.": "Uzun zaman önce yaptığınız hatalar hâlâ sizi rahatsız eder.",
    "You rarely contemplate the reasons for human existence or the meaning of life.": "İnsan varoluşunun nedenleri veya hayatın anlamı üzerine nadiren düşünürsünüz.",
    "Your emotions control you more than you control them.": "Duygularınız sizi, sizin onları kontrol ettiğinizden daha fazla kontrol eder.",
    "You take great care not to make people look bad, even when it is completely their fault.": "Tamamen onların hatası olsa bile insanları kötü göstermemeye büyük özen gösterirsiniz.",
    "Your personal work style is closer to spontaneous bursts of energy than organized and consistent efforts.": "Kişisel çalışma tarzınız düzenli ve istikrarlı çabadan çok spontane enerji patlamalarına yakındır.",
    "When someone thinks highly of you, you wonder how long it will take them to feel disappointed in you.": "Biri sizin hakkınızda iyi düşündüğünde, ne kadar sürede hayal kırıklığına uğrayacağını merak edersiniz.",
    "You would love a job that requires you to work alone most of the time.": "Çoğu zaman yalnız çalışmanızı gerektiren bir işi seversiniz.",
    "You believe that pondering abstract philosophical questions is a waste of time.": "Soyut felsefi sorular üzerine düşünmenin zaman kaybı olduğuna inanırsınız.",
    "You feel more drawn to places with busy, bustling atmospheres than quiet, intimate places.": "Sessiz ve samimi yerlerden çok hareketli, kalabalık atmosfere sahip yerlere çekilirsiniz.",
    "You know at first glance how someone is feeling.": "Birinin nasıl hissettiğini ilk bakışta anlarsınız.",
    "You often feel overwhelmed.": "Sık sık bunalmış hissedersiniz.",
    "You complete things methodically without skipping over any steps.": "Hiçbir adımı atlamadan işleri yöntemli şekilde tamamlarsınız.",
    "You are very intrigued by things labeled as controversial.": "Tartışmalı olarak nitelendirilen şeyler sizi çok cezbeder.",
    "You would pass along a good opportunity if you thought someone else needed it more.": "Başka birinin daha çok ihtiyacı olduğunu düşünürseniz iyi bir fırsatı ona bırakırsınız.",
    "You struggle with deadlines.": "Teslim tarihleriyle zorlanırsınız.",
    "You feel confident that things will work out for you.": "İşlerin sizin için yoluna gireceğinden eminsinizdir.",
}

PERSONALITY_DESCRIPTIONS = {
    "en": {
        "INTJ": "Architect",
        "INTP": "Logician",
        "ENTJ": "Commander",
        "ENTP": "Debater",
        "INFJ": "Advocate",
        "INFP": "Mediator",
        "ENFJ": "Protagonist",
        "ENFP": "Campaigner",
        "ISTJ": "Logistician",
        "ISFJ": "Defender",
        "ESTJ": "Executive",
        "ESFJ": "Consul",
        "ISTP": "Virtuoso",
        "ISFP": "Adventurer",
        "ESTP": "Entrepreneur",
        "ESFP": "Entertainer",
    },
    "tr": {
        "INTJ": "Architect",
        "INTP": "Logician",
        "ENTJ": "Commander",
        "ENTP": "Debater",
        "INFJ": "Advocate",
        "INFP": "Mediator",
        "ENFJ": "Protagonist",
        "ENFP": "Campaigner",
        "ISTJ": "Logistician",
        "ISFJ": "Defender",
        "ESTJ": "Executive",
        "ESFJ": "Consul",
        "ISTP": "Virtuoso",
        "ISFP": "Adventurer",
        "ESTP": "Entrepreneur",
        "ESFP": "Entertainer",
    },
}


def normalize_lang(lang):
    return lang if lang in TRANSLATIONS else "en"


def get_page_context(lang):
    lang = normalize_lang(lang)
    return {
        "lang": lang,
        "text": TRANSLATIONS[lang],
        "scale_labels": SCALE_LABELS[lang],
        "question_labels": QUESTION_LABELS_TR if lang == "tr" else {},
    }


@app.get("/")
def home(request: Request, lang: str = "en"):
    lang = normalize_lang(lang)

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            **get_page_context(lang),
            "features": features,
            "prediction": None,
            "description": None,
            "confidence": None,
            "top_3": None,
            "error": None,
            "selected_values": {},
            "confidence_note": None,
        },
    )


@app.post("/predict")
async def predict(request: Request, lang: str = "en"):
    lang = normalize_lang(lang)
    page_context = get_page_context(lang)
    text = TRANSLATIONS[lang]
    personality_descriptions = PERSONALITY_DESCRIPTIONS[lang]
    form_data = await request.form()

    try:
        input_dict = {}

        for feature in features:
            input_dict[feature] = int(form_data.get(feature))

        selected_values = input_dict

        input_df = pd.DataFrame([input_dict])
        input_df = input_df[features]

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        proba_df = (
            pd.DataFrame({
                "personality": model.classes_,
                "probability": probabilities,
            })
            .sort_values(by="probability", ascending=False)
            .head(3)
        )

        top_3 = [
            {
                "personality": row["personality"],
                "description": personality_descriptions.get(row["personality"], ""),
                "probability": round(row["probability"] * 100, 2),
            }
            for _, row in proba_df.iterrows()
        ]

        confidence = round(float(proba_df.iloc[0]["probability"]) * 100, 2)

        if confidence < 30:
            confidence_note = text["low_confidence"]
        elif confidence < 60:
            confidence_note = text["moderate_confidence"]
        else:
            confidence_note = text["high_confidence"]

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                **page_context,
                "features": features,
                "prediction": prediction,
                "description": personality_descriptions.get(prediction, ""),
                "confidence": confidence,
                "top_3": top_3,
                "error": None,
                "selected_values": selected_values,
                "confidence_note": confidence_note,
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                **page_context,
                "features": features,
                "prediction": None,
                "description": None,
                "confidence": None,
                "top_3": None,
                "error": str(e),
                "selected_values": {},
                "confidence_note": None,
            },
        )
