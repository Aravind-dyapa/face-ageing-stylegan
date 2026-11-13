
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Face Ageing Experiment Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Face Ageing — Experiment Results")
st.write(
    "Comparison of WNNR-inspired latent regularization "
    "at different lambda values."
)

st.info(
    "Experiment conditions: Target age = 60, same input face image. "
    "These are recorded measurements from the Streamlit experiments."
)

# Recorded experiment results
data = [
    {
        "Lambda": 0.5,
        "Nuclear Norm": 361.129,
        "SSIM": 0.99543,
        "PSNR (dB)": 42.73
    },
    {
        "Lambda": 1.0,
        "Nuclear Norm": 355.729,
        "SSIM": 0.98813,
        "PSNR (dB)": 37.13
    },
    {
        "Lambda": 2.0,
        "Nuclear Norm": 344.929,
        "SSIM": 0.97137,
        "PSNR (dB)": 31.90
    }
]

df = pd.DataFrame(data)

baseline_nuclear_norm = 366.529
df["Nuclear Norm Reduction (%)"] = (
    (baseline_nuclear_norm - df["Nuclear Norm"])
    / baseline_nuclear_norm
) * 100

# Summary metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Baseline Nuclear Norm", f"{baseline_nuclear_norm:.3f}")

with col2:
    st.metric(
        "Lowest Nuclear Norm",
        f"{df['Nuclear Norm'].min():.3f}"
    )

with col3:
    st.metric(
        "Highest SSIM",
        f"{df['SSIM'].max():.5f}"
    )

st.divider()

# Results table
st.subheader("1. Recorded Experiment Results")

display_df = df.copy()
display_df["Nuclear Norm Reduction (%)"] = (
    display_df["Nuclear Norm Reduction (%)"].round(2)
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "The baseline nuclear norm is from the unregularized "
    "age-60 experiment (lambda = 0)."
)

st.divider()

# Nuclear norm chart
st.subheader("2. Nuclear Norm Comparison")

norm_chart = df.set_index("Lambda")[["Nuclear Norm"]]
st.bar_chart(norm_chart)

st.caption(
    "Increasing lambda produces a lower nuclear norm "
    "in these recorded experiments."
)

st.divider()

# SSIM chart
st.subheader("3. SSIM Comparison")

ssim_chart = df.set_index("Lambda")[["SSIM"]]
st.bar_chart(ssim_chart)

st.divider()

# PSNR chart
st.subheader("4. PSNR Comparison")

psnr_chart = df.set_index("Lambda")[["PSNR (dB)"]]
st.bar_chart(psnr_chart)

st.divider()

# Interpretation
st.subheader("5. Observations")

st.markdown("""
- **Lambda = 0.5:** Highest SSIM and PSNR among the three regularized
  experiments, meaning its output is most similar to the baseline.
- **Lambda = 1.0:** Shows a stronger reduction in nuclear norm and a
  larger difference from the baseline.
- **Lambda = 2.0:** Produces the lowest nuclear norm and the greatest
  difference from the baseline among these settings.

These measurements describe changes relative to the baseline. They
do not, by themselves, establish better age accuracy, image quality,
or identity preservation.
""")

st.divider()

# CSV download
st.subheader("6. Download Experiment Data")

csv_data = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Results as CSV",
    data=csv_data,
    file_name="face_ageing_experiment_results.csv",
    mime="text/csv"
)

st.success("Experiment dashboard loaded successfully.")
