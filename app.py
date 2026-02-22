import streamlit as st
from pdfminer.high_level import extract_text
import spacy
from collections import Counter
import heapq
from fpdf import FPDF
import matplotlib.pyplot as plt
import re
import pandas as pd 

@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.warning("⚠️ spaCy model not found. Using basic text processing.")
        return None

nlp = load_spacy()

# Predefined risk-related words
RISK_WORDS = [
    "fraud", "penalty", "violation", "risk", "lawsuit", "breach",
    "noncompliance", "litigation", "regulatory", "fine"
]

def extract_text_from_pdf(uploaded_file):
    return extract_text(uploaded_file)

@st.cache_data
def process_text(text):
    if nlp is None:
        return None
    return nlp(text)

def extract_key_clauses(text):
    if nlp is None:
        # Fallback: split by sentences using basic punctuation
        sentences = re.split(r'[.!?]+', text)
        clauses = [s.strip() for s in sentences if len(s.strip()) > 10]
        return clauses[:10]
    else:
        doc = process_text(text)
        sentences = list(doc.sents)
        clauses = [str(sentence).strip() for sentence in sentences if len(sentence) > 10]
        return clauses[:10]

def summarize_text(text, num_sentences=5):
    if nlp is None:
        # Fallback: simple sentence extraction
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return '. '.join(sentences[:num_sentences])
    else:
        doc = process_text(text)
        sentences = list(doc.sents)
        word_frequencies = Counter([token.text.lower() for token in doc if token.is_alpha and not token.is_stop])
        sentence_scores = {sent: sum(word_frequencies.get(word.text.lower(), 0) for word in sent) for sent in sentences}
        summarized_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        return ' '.join([str(sentence) for sentence in summarized_sentences])

def detect_risks(text):
    if nlp is None:
        # Fallback: direct string matching
        text_lower = text.lower()
        return list(set(word for word in RISK_WORDS if word in text_lower))
    else:
        doc = process_text(text.lower())
        return list(set(token.text for token in doc if token.text in RISK_WORDS))

def extract_legal_entities(text):
    entities = {
        "PERSONS": [],
        "ORGANIZATIONS": [],
        "DATES": [],
        "MONETARY": [],
        "LEGAL_TERMS": []
    }
    
    if nlp is None:
        # Basic regex-based entity extraction
        import re
        
        # Find dates
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b'
        entities["DATES"] = list(set(re.findall(date_pattern, text)))
        
        # Find monetary amounts
        money_pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|dollars?|cents?)'
        entities["MONETARY"] = list(set(re.findall(money_pattern, text, re.IGNORECASE)))
        
        # Find legal terms
        legal_terms = ["contract", "agreement", "liability", "indemnity", "warranty", "termination", "jurisdiction", "governing law"]
        text_lower = text.lower()
        entities["LEGAL_TERMS"] = [term for term in legal_terms if term in text_lower]
        
    else:
        doc = process_text(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities["PERSONS"].append(ent.text)
            elif ent.label_ == "ORG":
                entities["ORGANIZATIONS"].append(ent.text)
            elif ent.label_ == "DATE":
                entities["DATES"].append(ent.text)
            elif ent.label_ == "MONEY":
                entities["MONETARY"].append(ent.text)
    
    # Remove duplicates and limit results
    for key in entities:
        entities[key] = list(set(entities[key]))[:10]
    
    return entities

def check_compliance(text):
    compliance_issues = []
    text_lower = text.lower()
    
    # Check for common compliance issues
    if "governing law" not in text_lower and "jurisdiction" not in text_lower:
        compliance_issues.append("Missing governing law or jurisdiction clause")
    
    if "termination" not in text_lower:
        compliance_issues.append("No termination clause found")
    
    if "confidential" not in text_lower and "proprietary" not in text_lower:
        compliance_issues.append("No confidentiality or proprietary information clause")
    
    if "liability" not in text_lower:
        compliance_issues.append("Liability terms not clearly defined")
    
    # Check for signature requirements
    if "signature" not in text_lower and "signed" not in text_lower:
        compliance_issues.append("Document may lack proper signature requirements")
    
    return compliance_issues

def generate_legal_report(summary, clauses, risks, entities, compliance_issues, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Legal Document Analysis Report", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Document: {filename}", ln=True)
    pdf.cell(0, 10, f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(10)
    
    # Executive Summary
    if summary:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Executive Summary:", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 8, summary)
        pdf.ln(5)
    
    # Key Legal Clauses
    if clauses:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Key Legal Clauses:", ln=True)
        pdf.set_font("Arial", "", 10)
        for i, clause in enumerate(clauses, 1):
            pdf.multi_cell(0, 8, f"{i}. {clause}")
        pdf.ln(5)
    
    # Risk Assessment
    if risks:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Risk Assessment:", ln=True)
        pdf.set_font("Arial", "", 10)
        for risk in risks:
            pdf.cell(0, 8, f"• {risk.title()}", ln=True)
        pdf.ln(5)
    
    # Entity Recognition
    if entities:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Identified Entities:", ln=True)
        pdf.set_font("Arial", "", 10)
        for entity_type, entity_list in entities.items():
            if entity_list:
                pdf.cell(0, 8, f"{entity_type}: {', '.join(entity_list)}", ln=True)
        pdf.ln(5)
    
    # Compliance Issues
    if compliance_issues:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Compliance Issues:", ln=True)
        pdf.set_font("Arial", "", 10)
        for issue in compliance_issues:
            pdf.cell(0, 8, f"⚠ {issue}", ln=True)
        pdf.ln(5)
    
    pdf_path = f"Legal_Analysis_{filename.replace('.pdf', '')}.pdf"
    pdf.output(pdf_path)
    return pdf_path

def get_regulatory_updates():
    predefined_updates = [
        {"title": "📜 New Compliance Guidelines", "summary": "SEC released new guidelines for regulatory compliance."},
        {"title": "⚖️ Update on Financial Risks", "summary": "New policies to mitigate risks in the financial sector."},
    ]
    return predefined_updates

def visualize_key_clauses_frequency(clauses):
    clause_counts = Counter(clauses)
    common_clauses = clause_counts.most_common()
    if common_clauses:
        labels, values = zip(*common_clauses)
        plt.figure(figsize=(10, 6))
        plt.barh(labels, values, color='skyblue')
        plt.xlabel('Frequency')
        plt.title('📊 Key Clauses Frequency')
        st.pyplot(plt)
    else:
        st.write("🚫 No key clauses to visualize.")

def generate_pdf_report(summary, clauses, risks, updates):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Legal Document Analysis Report", ln=True, align="C")
    pdf.ln(10)
    
    if summary:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Summary:", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 10, summary)
        pdf.ln(5)
    
    if clauses:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Key Clauses:", ln=True)
        pdf.set_font("Arial", "", 10)
        for i, clause in enumerate(clauses, 1):
            pdf.multi_cell(0, 10, f"{i}. {clause}")
        pdf.ln(5)
    
    if risks:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Detected Risks:", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 10, ", ".join(risks))
        pdf.ln(5)
    
    if updates:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Regulatory Updates:", ln=True)
        pdf.set_font("Arial", "", 10)
        for update in updates:
            pdf.multi_cell(0, 10, f"- {update.get('title')}: {update.get('summary')}")
    
    pdf_path = "Analysis_Results.pdf"
    pdf.output(pdf_path)
    return pdf_path

def main():
    st.title("⚖️ Legal Document NLP Toolkit")
    st.markdown("### 📋 Advanced Legal Document Analysis & Summarization Platform")
    
    st.sidebar.title("🔧 NLP Toolkit Options")
    features = st.sidebar.multiselect("🔍 Select Analysis Features", 
                                       ["� Document Summary", "🔑 Key Legal Clauses", "⚖️ Risk Assessment", 
                                        "📊 Entity Recognition", "🎯 Compliance Check", "📈 Data Visualization"])
    
    uploaded_file = st.file_uploader("📂 Upload Legal Document (PDF)", type="pdf", 
                                  help="Upload contracts, agreements, or legal documents for analysis")
    
    # Add document info section
    if uploaded_file is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(f"📄 **Document:** {uploaded_file.name}")
        with col2:
            file_size = len(uploaded_file.getvalue()) / 1024
            st.info(f"📊 **Size:** {file_size:.1f} KB")
    
    if uploaded_file is not None:
        try:
            text = extract_text_from_pdf(uploaded_file)
            st.success("✅ Legal document processed successfully!")
            
            # Document statistics
            with st.expander("📊 Document Statistics", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📝 Words", len(text.split()))
                with col2:
                    st.metric("📄 Characters", len(text))
                with col3:
                    st.metric("🔤 Sentences", len(re.split(r'[.!?]+', text)))
                with col4:
                    st.metric("📖 Paragraphs", len(text.split('\n\n')))
                    
        except Exception as e:
            st.error(f"❌ Error processing legal document: {e}")
            return

        summary, clauses, risks, entities, compliance_issues = "", [], [], [], []

        if "� Document Summary" in features:
            with st.spinner("🤖 Generating AI-powered summary..."):
                summary = summarize_text(text)
            st.subheader("� Executive Summary")
            st.write(summary)
            st.info("💡 **Summary Method:** " + ("Advanced NLP with spaCy" if nlp else "Basic text extraction"))

        if "🔑 Key Legal Clauses" in features:
            with st.spinner("⚖️ Extracting legal clauses..."):
                clauses = extract_key_clauses(text)
            st.subheader("🔑 Key Legal Clauses")
            for i, clause in enumerate(clauses, 1):
                st.write(f"**{i}.** {clause}")
            
            if "� Data Visualization" in features and clauses:
                visualize_key_clauses_frequency(clauses)

        if "⚖️ Risk Assessment" in features:
            with st.spinner("🔍 Analyzing legal risks..."):
                risks = detect_risks(text)
            st.subheader("⚖️ Risk Assessment Report")
            if risks:
                risk_colors = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                for risk in risks:
                    risk_level = "medium" if risk in ["violation", "breach"] else "low"
                    st.write(f"{risk_colors.get(risk_level, '⚪')} **{risk.title()}** - {risk_level.upper()} priority")
            else:
                st.success("✅ No significant legal risks detected")

        if "🎯 Entity Recognition" in features:
            with st.spinner("🏷️ Identifying legal entities..."):
                entities = extract_legal_entities(text)
            st.subheader("🏷️ Legal Entity Recognition")
            if entities:
                for entity_type, entity_list in entities.items():
                    if entity_list:
                        st.write(f"**{entity_type}:** {', '.join(entity_list[:5])}")
            else:
                st.info("ℹ️ No specific legal entities identified")

        if "🎯 Compliance Check" in features:
            with st.spinner("✅ Checking compliance requirements..."):
                compliance_issues = check_compliance(text)
            st.subheader("🎯 Compliance Analysis")
            if compliance_issues:
                for issue in compliance_issues:
                    st.warning(f"⚠️ {issue}")
            else:
                st.success("✅ Document appears compliant with standard legal requirements")

        # Generate comprehensive report
        if features:
            st.markdown("---")
            if st.button("� Generate Comprehensive Legal Report"):
                pdf_path = generate_legal_report(summary, clauses, risks, entities, compliance_issues, uploaded_file.name)
                st.success("📥 Legal Analysis Report Ready!")
                with open(pdf_path, "rb") as file:
                    st.download_button("📥 Download Legal Report", file, 
                                    file_name=f"Legal_Analysis_{uploaded_file.name}.pdf", 
                                    mime="application/pdf")

if __name__ == "__main__":
    main()

