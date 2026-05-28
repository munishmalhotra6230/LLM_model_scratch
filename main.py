import streamlit as st
from spam_ham_classifier.predict import Model, classify_review, tokenizer, device
import json

st.set_page_config(
    page_title="Quasar Spam/Ham Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished UI
_CSS = """
<style>
body { background: linear-gradient(180deg, #f7fbff 0%, #ffffff 60%); color: #000000; }
* { color: #000000; }
.card { background: white; padding: 20px; border-radius: 14px; 
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06); margin: 12px 0; color: #000000; }
h1, h2, h3, h4, h5, h6, p, li, div { color: #000000; }
.brand { font-weight: 700; color: #0b74de; font-size: 28px; }
.muted { color: #505050; font-size: 14px; }
.result-spam { color: #000000; font-weight: 700; font-size: 24px; }
.result-clean { color: #047857; font-weight: 700; font-size: 24px; }
.metric-box { background: #f3f4f6; padding: 14px; border-radius: 10px; margin: 8px 0; color: #000000; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# Header
st.markdown("""
<div class="card">
  <div class="brand">🚀 Quasar Spam/Ham Classifier</div>
  <div class="muted">A transformer-based spam detection system built from scratch</div>
</div>
""", unsafe_allow_html=True)

# Main layout
col1, col2 = st.columns([2, 1], gap="medium")

with col1:
    st.markdown("""
    <div class="card">
      <h3>Project Overview</h3>
      <p><strong>Quasar</strong> is a lightweight GPT-style transformer model built entirely from scratch 
      and fine-tuned for binary spam classification. Every component—multi-head attention, layer normalization, 
      GELU activation, feed-forward blocks—was implemented from the ground up without relying on pre-built 
      modules for these core layers.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <h4>🔧 Model Architecture</h4>
      <ul>
        <li><strong>Vocabulary Size:</strong> 50,257 (GPT-2 tokenizer)</li>
        <li><strong>Context Length:</strong> 1,024 tokens</li>
        <li><strong>Embedding Dimension:</strong> 768</li>
        <li><strong>Attention Heads:</strong> 12</li>
        <li><strong>Transformer Layers:</strong> 12</li>
        <li><strong>Tokenizer:</strong> GPT-2 BPE</li>
        <li><strong>Device:</strong> {}</li>
      </ul>
      <p class="muted"><strong>Training:</strong> Model was trained on GPU in Google Colab with a custom 
      spam/ham dataset. The original language model head was replaced with a 2-class classification head 
      for binary classification. Refer to the notebook files for training details.</p>
    </div>
    """.format(device), unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <h4>📝 How It Works</h4>
      <ol>
        <li>Enter a message in the input below</li>
        <li>The message is tokenized using GPT-2 BPE tokenizer</li>
        <li>Tokens are passed through 12 transformer blocks with multi-head attention</li>
        <li>The final token's hidden state is passed to the classifier head</li>
        <li>Output probabilities for spam (1) and not-spam (0) are computed</li>
      </ol>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
      <h4>📚 Resources</h4>
      <p class="muted">This project includes:</p>
      <ul>
        <li><strong>model.py</strong> — Full transformer architecture</li>
        <li><strong>making_llm_from_scratch/</strong> — Training notebooks</li>
        <li><strong>predict.py</strong> — Inference utilities</li>
        <li><strong>dashbpard.py</strong> — Interactive dashboard</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <h4>💡 Key Features</h4>
      <ul>
        <li>Built from scratch</li>
        <li>No pre-built attention modules</li>
        <li>Custom layer normalization</li>
        <li>Finetuned on labeled data</li>
        <li>Fast inference on CPU/GPU</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

# Separator
st.markdown("---")

# Classifier section
st.markdown("""
<div class="card">
  <h3>🔍 Message Classifier</h3>
  <p class="muted">Paste or type a message to classify as spam or not spam.</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state for results
if 'pred_result' not in st.session_state:
    st.session_state.pred_result = None
if 'pred_prob' not in st.session_state:
    st.session_state.pred_prob = None

with st.form(key="classify_form"):
    text = st.text_area("Message", placeholder="Enter a message...", height=120)
    submitted = st.form_submit_button("Classify", use_container_width=True)

if submitted:
    if not text.strip():
        st.warning("⚠️ Please enter a message to classify.")
    else:
        with st.spinner("Analyzing message..."):
            try:
                result, prob = classify_review(text, model=Model, tokenizer=tokenizer, device=device)
                st.session_state.pred_result = result
                st.session_state.pred_prob = prob
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.session_state.pred_result = None
                st.session_state.pred_prob = None

# Display prediction results if they exist
if st.session_state.pred_result:
    st.markdown("""
    <div class="card">
      <h4>Prediction Result</h4>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.pred_result == 'spam':
        st.markdown(f'<div class="result-spam">🚨 SPAM DETECTED</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-clean">✅ NOT SPAM</div>', unsafe_allow_html=True)

    # Probabilities
    if st.session_state.pred_prob and len(st.session_state.pred_prob) >= 2:
        not_spam_prob = st.session_state.pred_prob[0]
        spam_prob = st.session_state.pred_prob[1]

        st.markdown("""
        <div class="card">
          <h4 style="color: #000; margin-bottom: 20px;">📊 Classification Confidence</h4>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div style="background:#ffffff;padding:12px;border-radius:8px;text-align:center;">
              <div style="font-size:14px;color:#333;">✅ Not Spam</div>
              <div style="font-size:28px;font-weight:700;color:#000;">{not_spam_prob*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div style="background:#ffffff;padding:12px;border-radius:8px;text-align:center;">
              <div style="font-size:14px;color:#333;">⚠️ Spam</div>
              <div style="font-size:28px;font-weight:700;color:#b91c1c;">{spam_prob*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.progress(min(max(spam_prob, 0.0), 1.0))

        # Use Streamlit-native boxes so values remain visible in both light/dark themes
        st.write("📈 **Raw Probability Values:**")
        st.info(f"🚫 Not Spam: {not_spam_prob:.15f}")
        st.warning(f"⚠️ Spam: {spam_prob:.15f}")

st.markdown("---")
st.markdown(
    "<p class='muted' style='text-align:center'>Built with ❤️ using PyTorch, Streamlit, and custom transformer architecture</p>",
    unsafe_allow_html=True
)
