import streamlit as st
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
import openai
import io

# --- Konfigurasi Halaman & Judul ---
st.set_page_config(page_title="Sales Forecast AI Analyst", layout="wide")

st.title("📈 Sales Forecast & AI Analyst")
st.markdown("""
Aplikasi ini menggunakan model Prophet untuk membuat peramalan penjualan dan mengintegrasikan GPT untuk memberikan wawasan bisnis.
**Cara Penggunaan:**
1.  Unggah file CSV data penjualan Anda atau gunakan data contoh.
2.  Sesuaikan periode peramalan dan klik 'Run Forecast'.
3.  *Catatan: Admin aplikasi ini telah mengkonfigurasi API Key di backend.*
""")

# --- Sidebar untuk Input Pengguna ---
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    
    # Opsi Unggah File
    st.subheader("Unggah Data Penjualan")
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
    st.markdown("""
    *File CSV harus memiliki 2 kolom:*
    - `ds` (tanggal, format: YYYY-MM-DD)
    - `y` (nilai penjualan, numerik)
    """)

    # Opsi Periode Peramalan
    st.subheader("Parameter Peramalan")
    periods_input = st.number_input('Jumlah hari untuk peramalan', min_value=30, max_value=365, value=90)

    # Tombol untuk Menjalankan Proses
    run_button = st.button("🚀 Run Forecast & Analysis")

# --- Fungsi untuk Analisis AI ---
def get_sales_insights(forecast_df):
    """Menggunakan GPT untuk menganalisis data peramalan dan memberikan wawasan."""
    # Mengambil API key dari st.secrets, bukan dari input pengguna
    if "openai" not in st.secrets or "api_key" not in st.secrets.openai:
        return "Konfigurasi OpenAI API Key belum diatur oleh admin aplikasi di st.secrets."
    
    try:
        openai.api_key = st.secrets.openai.api_key
        
        # Menyiapkan data untuk prompt
        prompt_data = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).to_string()

        prompt = f"""
        Anda adalah seorang analis bisnis senior yang ahli dalam interpretasi data penjualan.
        Berdasarkan data peramalan penjualan berikut untuk 30 hari ke depan:
        {prompt_data}

        Berikan analisis dan rekomendasi strategis dalam format berikut:

        **1. Ringkasan Eksekutif:**
        Berikan ringkasan singkat (2-3 kalimat) tentang tren penjualan yang diperkirakan.

        **2. Identifikasi Tren & Pola:**
        - Apakah tren utamanya (naik, turun, atau stabil)?
        - Apakah ada pola musiman atau mingguan yang terlihat dari data?
        - Kapan perkiraan puncak dan lembah penjualan terjadi?

        **3. Potensi Risiko & Peluang:**
        - Identifikasi potensi risiko (misalnya, penurunan penjualan yang tajam).
        - Identifikasi peluang (misalnya, periode penjualan puncak yang bisa dimaksimalkan).

        **4. Rekomendasi Aksi:**
        - **Untuk Tim Pemasaran:** Berikan 2-3 saran konkret (misalnya, kapan harus meluncurkan promosi).
        - **Untuk Tim Inventaris:** Berikan 2-3 saran tentang manajemen stok berdasarkan peramalan.
        """

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a senior business analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Terjadi kesalahan saat menghubungi API OpenAI: {e}"


# --- Logika Utama Aplikasi ---
if run_button:
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Validasi kolom
            if 'ds' not in df.columns or 'y' not in df.columns:
                st.error("Error: File CSV harus memiliki kolom 'ds' dan 'y'.")
            else:
                df['ds'] = pd.to_datetime(df['ds'])
                st.session_state.df = df
        except Exception as e:
            st.error(f"Gagal memuat file: {e}")
    elif 'df' not in st.session_state:
        # Gunakan data contoh jika tidak ada file yang diunggah
        st.info("Tidak ada file diunggah. Menggunakan data contoh.")
        # Membuat data contoh
        data = {
            'ds': pd.to_datetime(pd.date_range('2023-01-01', periods=365)),
            'y': [100 + i/5 + 20 * (1 + pd.Timestamp(d).dayofweek // 5) + 30 * (1 + pd.Timestamp(d).month // 3) for i, d in enumerate(pd.date_range('2023-01-01', periods=365))]
        }
        st.session_state.df = pd.DataFrame(data)

    if 'df' in st.session_state:
        df = st.session_state.df
        
        with st.spinner('Menjalankan model Prophet...'):
            m = Prophet()
            m.fit(df)
            future = m.make_future_dataframe(periods=periods_input)
            forecast = m.predict(future)
            st.session_state.forecast = forecast
            st.session_state.model = m
        
        st.success('Peramalan selesai!')

# --- Tampilkan Hasil ---
if 'forecast' in st.session_state:
    forecast = st.session_state.forecast
    m = st.session_state.model
    df = st.session_state.df

    tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Analyst Insights", "📊 Forecast Chart", "📈 Forecast Components", "📄 Data"])

    with tab1:
        st.header("Analisis & Rekomendasi dari AI")
        with st.spinner("AI sedang menganalisis data..."):
            insights = get_sales_insights(forecast)
            st.markdown(insights)

    with tab2:
        st.header("Visualisasi Peramalan Penjualan")
        fig1 = plot_plotly(m, forecast)
        fig1.update_layout(title='Peramalan Penjualan vs Data Aktual', xaxis_title='Tanggal', yaxis_title='Penjualan')
        st.plotly_chart(fig1, use_container_width=True)

    with tab3:
        st.header("Komponen Model Peramalan")
        fig2 = plot_components_plotly(m, forecast)
        st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        st.header("Data Peramalan")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods_input + 5))
        
        csv = forecast.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Forecast Data as CSV",
            data=csv,
            file_name='sales_forecast.csv',
            mime='text/csv',
        )
