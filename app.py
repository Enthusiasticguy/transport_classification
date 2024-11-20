import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# Sarlavha va uslub
st.set_page_config(page_title="Transportni Klassifikatsiya", page_icon="🚗")
st.title('🚀 Transportni Klassifikatsiya qiluvchi Model')
st.markdown("""
    **Samolyotmi? Mashinami? Qayiqmi?**   
    Ushbu ilova yuklangan rasmni klassifikatsiya qiladi va natijani ehtimollik bilan ko'rsatadi.   
    Yuklang va sinab ko'ring! 😎
""")

# Rasm yuklash
file = st.file_uploader(
    'Rasm yuklang (faqat *qayiq*, *samolyot*, yoki *mashina*)',
    type=['png', 'jpeg', 'jpg', 'gif']
)

# Modelni yuklash
model = load_learner('transport_model.pkl')

# Rasmni tahlil qilish
if file:
    st.image(file, caption="Yuklangan rasm", use_container_width=True)

    with st.spinner("🔍 Bashorat qilinmoqda..."):
        img = PILImage.create(file)
        pred, pred_id, probs = model.predict(img)

    st.success(f'**Bashorat**: {pred}')
    st.info(f'**Ehtimollik**: {probs[pred_id] * 100:.1f}%')

    # Har bir transport uchun qisqa izoh
    descriptions = {
        "car": "Bu to'g'ri bashorat! Mashinalar shahar va uzoq masofalarga qulay transport vositasidir.",
        "boat": "Bu qayiqmi? Suv bo'yida foydalanish uchun ideal tanlov!",
        "plane": "Samolyot! Osmonni zabt etish uchun ajoyib texnologiya."
    }
    st.write(f"**Ma'lumot:** {descriptions.get(pred, 'Nomalum transport vositasi.')}")

    # Bar grafigi
    fig = px.bar(
        x=model.dls.vocab,
        y=probs * 100,
        color=model.dls.vocab,
        labels={"x": "Transport turi", "y": "Ehtimollik (%)"},
        title="Ehtimolliklar taqsimoti",
        template="plotly_dark"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)
else:
    st.info("❗️ Rasm yuklang va natijani ko'ring.")
st.info('https://t.me/ismoilov_husan')
