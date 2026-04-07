import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF
from openai import OpenAI


st.set_page_config(
    page_title="Transfer Pricing AI Reasoning Engine",
    page_icon=":bar_chart:",
    layout="wide",
)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
            .main {
                background: linear-gradient(180deg, #f8fbff 0%, #eef5ff 100%);
            }
            h1, h2, h3 {
                color: #0c2d6b;
            }
            [data-testid="stSidebar"] {
                background: #0f1f3d;
            }
            [data-testid="stSidebar"] * {
                color: #f5f8ff !important;
            }
            .disclaimer {
                padding: 0.6rem 0.8rem;
                border-left: 4px solid #1f5bd7;
                background: #eaf1ff;
                border-radius: 6px;
                color: #163870;
                font-size: 0.92rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def resolve_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    norm_map = {normalize_name(col): col for col in df.columns}
    for alias in aliases:
        key = normalize_name(alias)
        if key in norm_map:
            return norm_map[key]
    return None


def privacy_shield(text: str) -> str:
    sanitized = re.sub(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", "[REDACTED_PAN]", text)
    sanitized = re.sub(r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z]\b", "[REDACTED_GSTIN]", sanitized)
    sanitized = re.sub(r"\b[L|U][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6}\b", "[REDACTED_CIN]", sanitized)
    sanitized = re.sub(r"\b\d{10,16}\b", "[REDACTED_ID]", sanitized)
    return sanitized


REJECTION_BUCKETS = [
    "Functional Dissimilarity",
    "Economic/Asset Profile",
    "Risk Profile",
    "Quantitative Thresholds",
    "Extraordinary Events",
]


SYSTEM_PROMPT = (
    "You are a Senior Transfer Pricing Partner at a Tier-1 Indian Law Firm. "
    "Your language must be formal, authoritative, and cite ITAT-style logic regarding functional comparability "
    "where applicable. "
    "You must classify every rejection into exactly one bucket from the approved taxonomy."
)


def ask_llm(
    client: OpenAI,
    tested_party_far: str,
    business_description: str,
    extra_context: str,
) -> Dict[str, Any]:
    taxonomy = "; ".join(REJECTION_BUCKETS)
    prompt = f"""
Compare the following comparable company's business description with the tested party's FAR profile for Indian transfer pricing documentation.

Decision rule (mandatory): If the company is an IP-owner/product-developer and the client is a service-provider (routine captive), REJECT.

You must return strict JSON with exactly these keys:
- status: "Accept" or "Reject"
- bucket: one of [{taxonomy}] (required if status="Reject"; must be "N/A" if status="Accept")
- reason: 1 sentence, formal legal justification (Rule 10B / ITAT-style reasoning where applicable)
- reasoning_chain: if status="Accept", provide a short, bullet-like string linking 2-4 specific keywords/phrases found in the comparable text to the client's FAR; else empty string
- matched_keywords: if status="Accept", an array of 2-6 keywords/phrases copied verbatim from the comparable description or context; else []

Client FAR Analysis (sanitized):
{tested_party_far}

Comparable text (sanitized):
{business_description}

Additional context (sanitized; may include extraordinary events / risk indicators / ratios):
{extra_context}
""".strip()
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)
    return parsed


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def load_excel_sheets(file) -> Tuple[pd.ExcelFile, List[str]]:
    xls = pd.ExcelFile(file)
    sheets = list(xls.sheet_names)
    if not sheets:
        raise ValueError("No sheets found in the uploaded Excel.")
    return xls, sheets


def merge_comparables_and_ratios(comparables_df: pd.DataFrame, ratios_df: pd.DataFrame) -> pd.DataFrame:
    comp_company = resolve_column(comparables_df, ["Company Name", "Name", "Company", "CompanyName"])
    comp_desc = resolve_column(comparables_df, ["Business Description", "Business Profile", "Description"])
    if comp_company is None or comp_desc is None:
        raise ValueError("Comparables sheet must include 'Company Name' and 'Business Description' (or close aliases).")

    ratio_company = resolve_column(ratios_df, ["Company Name", "Name", "Company", "CompanyName"])
    rpt_col = resolve_column(ratios_df, ["RPT %", "RPT Percentage", "Related Party Transactions %", "Related Party Transactions"])
    margin_col = resolve_column(ratios_df, ["Operating Profit Margin", "OPM", "OP/OC", "Operating Margin", "Margin"])
    turnover_col = resolve_column(ratios_df, ["Turnover", "Sales", "Revenue", "Operating Revenue", "Total Revenue"])
    turnover_growth_col = resolve_column(ratios_df, ["Turnover Growth %", "Sales Growth %", "Revenue Growth %", "YoY Growth %", "Growth %"])
    extraordinary_col = resolve_column(
        comparables_df, ["Extraordinary Events", "Merger", "Acquisition", "Demerger", "Amalgamation", "Exceptional Items"]
    ) or resolve_column(ratios_df, ["Extraordinary Events", "Merger", "Acquisition", "Demerger", "Amalgamation", "Exceptional Items"])

    required_ratios = {"Company Name": ratio_company, "RPT %": rpt_col, "Operating Profit Margin": margin_col}
    missing = [k for k, v in required_ratios.items() if v is None]
    if missing:
        raise ValueError(f"Financial ratios sheet missing required columns: {', '.join(missing)}")

    comp = comparables_df[[comp_company, comp_desc] + ([extraordinary_col] if extraordinary_col and extraordinary_col in comparables_df.columns else [])].copy()
    comp.columns = ["Company Name", "Business Description"] + (["Extraordinary Events"] if extraordinary_col and extraordinary_col in comparables_df.columns else [])

    ratios_keep = [ratio_company, rpt_col, margin_col]
    rename_map = {ratio_company: "Company Name", rpt_col: "RPT %", margin_col: "Margin"}
    if turnover_col:
        ratios_keep.append(turnover_col)
        rename_map[turnover_col] = "Turnover"
    if turnover_growth_col:
        ratios_keep.append(turnover_growth_col)
        rename_map[turnover_growth_col] = "Turnover Growth %"
    if extraordinary_col and extraordinary_col in ratios_df.columns:
        ratios_keep.append(extraordinary_col)
        rename_map[extraordinary_col] = "Extraordinary Events"

    ratios = ratios_df[ratios_keep].copy().rename(columns=rename_map)

    merged = comp.merge(ratios, on="Company Name", how="left")
    merged["RPT %"] = to_numeric(merged["RPT %"])
    merged["Margin"] = to_numeric(merged["Margin"])
    if "Turnover" in merged.columns:
        merged["Turnover"] = to_numeric(merged["Turnover"])
    if "Turnover Growth %" in merged.columns:
        merged["Turnover Growth %"] = to_numeric(merged["Turnover Growth %"])
    if "Extraordinary Events" not in merged.columns:
        merged["Extraordinary Events"] = ""
    merged["Extraordinary Events"] = merged["Extraordinary Events"].astype(str).fillna("")
    return merged


def weighted_average(margins: pd.Series, weights: Optional[pd.Series]) -> Optional[float]:
    m = margins.dropna()
    if m.empty:
        return None
    if weights is None:
        return float(m.mean())
    w = weights.loc[m.index].astype(float)
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if float(w.sum()) <= 0:
        return float(m.mean())
    return float((m * w).sum() / w.sum())


def normalize_bucket(bucket: str) -> str:
    b = (bucket or "").strip()
    for allowed in REJECTION_BUCKETS:
        if normalize_name(b) == normalize_name(allowed):
            return allowed
    return "Quantitative Thresholds"


def build_pdf(
    results: pd.DataFrame,
    alr_low: Optional[float],
    alr_high: Optional[float],
    far_analysis: str,
    prowess_keywords: str,
    qualitative_criteria: List[str],
    weighted_avg: Optional[float],
    safe_harbor_flag: Optional[bool],
    tested_party_margin: Optional[float],
) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Benchmarking Memorandum (Rule 10B - Income-tax Rules, 1962)", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", ln=True)
    pdf.ln(2)
    pdf.multi_cell(0, 6, "Disclaimer: Data processed in-memory. Zero-retention policy active.")
    pdf.ln(2)

    accepted = int((results["Status"] == "Accept").sum())
    rejected = int((results["Status"] == "Reject").sum())
    pdf.cell(0, 8, f"Accepted comparables: {accepted} | Rejected comparables: {rejected}", ln=True)
    if alr_low is not None and alr_high is not None:
        pdf.cell(0, 8, f"Final Arm's Length Range (35th-65th): {alr_low:.2f}% - {alr_high:.2f}%", ln=True)
    else:
        pdf.cell(0, 8, "Final Arm's Length Range (35th-65th): Not available", ln=True)
    if weighted_avg is not None:
        pdf.cell(0, 8, f"Weighted average margin (final set): {weighted_avg:.2f}%", ln=True)
    if tested_party_margin is not None and safe_harbor_flag is not None and weighted_avg is not None:
        sh = "YES" if safe_harbor_flag else "NO"
        pdf.cell(
            0,
            8,
            f"Safe harbor check (±3% vs weighted average): {sh} (Tested party: {tested_party_margin:.2f}%)",
            ln=True,
        )
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1. Profile of the Tested Party (FAR Analysis)", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, far_analysis.strip() or "Not provided.")
    pdf.ln(1)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "2. Search Methodology (Prowess/Capitaline)", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, ("Keywords used:\n" + prowess_keywords.strip()) if prowess_keywords.strip() else "Keywords used: Not provided.")
    pdf.ln(1)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "3. Qualitative Screening Criteria Applied (Rule 10B - Comparability Factors)", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for c in qualitative_criteria:
        pdf.multi_cell(0, 6, f"- {c}")
    pdf.ln(1)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "4. Annexure: Detailed Accept/Reject Matrix with Tax-Technical Justifications", ln=True)
    pdf.ln(1)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(64, 8, "Company Name", border=1)
    pdf.cell(20, 8, "Margin", border=1)
    pdf.cell(22, 8, "Status", border=1)
    pdf.cell(30, 8, "Bucket", border=1)
    pdf.cell(0, 8, "Reason", border=1, ln=True)
    pdf.set_font("Helvetica", "", 9)

    for _, row in results.iterrows():
        company = str(row["Company Name"])[:35]
        margin = f"{row['Margin']:.2f}%" if pd.notna(row["Margin"]) else "NA"
        status = str(row["Status"])
        bucket = str(row.get("Bucket", ""))[:20]
        reason = str(row["Reason"])[:90]
        pdf.cell(64, 7, company, border=1)
        pdf.cell(20, 7, margin, border=1)
        pdf.cell(22, 7, status, border=1)
        pdf.cell(30, 7, bucket, border=1)
        pdf.cell(0, 7, reason, border=1, ln=True)

    return bytes(pdf.output(dest="S"))


def process_data(
    merged_df: pd.DataFrame,
    client: OpenAI,
    far_analysis: str,
    tested_party_margin: Optional[float],
) -> Tuple[pd.DataFrame, Optional[float], Optional[float], Optional[float], Optional[bool], str]:
    required_cols = ["Company Name", "RPT %", "Margin", "Business Description"]
    missing = [c for c in required_cols if c not in merged_df.columns]
    if missing:
        raise ValueError(f"Merged dataset missing required columns: {', '.join(missing)}")

    working = merged_df.copy()
    working["Status"] = "Pending"
    working["Reason"] = ""
    working["Bucket"] = ""
    working["Reasoning Chain"] = ""
    working["Matched Keywords"] = ""

    rpt_reject = working["RPT %"] > 25
    loss_reject = working["Margin"] < 0
    decline_reject: Optional[pd.Series] = None
    if "Turnover Growth %" in working.columns:
        decline_reject = working["Turnover Growth %"] < 0

    def apply_phase1(mask: pd.Series, reason: str) -> None:
        working.loc[mask, "Status"] = "Reject"
        working.loc[mask, "Bucket"] = "Quantitative Thresholds"
        working.loc[mask, "Reason"] = reason

    apply_phase1(rpt_reject, "Rejected (Quantitative Thresholds): Related Party Transactions exceed 25% threshold.")
    apply_phase1(loss_reject & (working["Status"] != "Reject"), "Rejected (Quantitative Thresholds): Persistent loss maker (operating margin below 0).")
    if decline_reject is not None:
        apply_phase1(
            decline_reject & (working["Status"] != "Reject"),
            "Rejected (Quantitative Thresholds): Declining turnover indicated in the financial ratios.",
        )

    for idx, row in working[working["Status"] == "Pending"].iterrows():
        clean_far = privacy_shield(far_analysis)
        clean_desc = privacy_shield(safe_str(row.get("Business Description")))
        extra_context = {
            "RPT %": safe_str(row.get("RPT %")),
            "Margin": safe_str(row.get("Margin")),
            "Turnover": safe_str(row.get("Turnover")),
            "Turnover Growth %": safe_str(row.get("Turnover Growth %")),
            "Extraordinary Events": safe_str(row.get("Extraordinary Events")),
        }
        clean_ctx = privacy_shield(json.dumps(extra_context, ensure_ascii=False))
        verdict = ask_llm(client, clean_far, clean_desc, clean_ctx)

        status = str(verdict.get("status", "Reject")).strip().title()
        if status not in {"Accept", "Reject"}:
            status = "Reject"

        if status == "Reject":
            bucket = normalize_bucket(str(verdict.get("bucket", "")))
            reason = safe_str(verdict.get("reason")) or "Rejected: Not comparable on Rule 10B qualitative criteria."
            working.at[idx, "Status"] = "Reject"
            working.at[idx, "Bucket"] = bucket
            working.at[idx, "Reason"] = reason
            working.at[idx, "Reasoning Chain"] = ""
            working.at[idx, "Matched Keywords"] = ""
        else:
            reason = safe_str(verdict.get("reason")) or "Accepted: Comparable on functional and risk profile."
            chain = safe_str(verdict.get("reasoning_chain"))
            keywords = verdict.get("matched_keywords", [])
            if not isinstance(keywords, list):
                keywords = []
            keywords_str = ", ".join([safe_str(k).strip() for k in keywords if safe_str(k).strip()])
            working.at[idx, "Status"] = "Accept"
            working.at[idx, "Bucket"] = "N/A"
            working.at[idx, "Reason"] = reason
            working.at[idx, "Reasoning Chain"] = chain
            working.at[idx, "Matched Keywords"] = keywords_str

    output = working[
        [
            "Company Name",
            "Margin",
            "Status",
            "Bucket",
            "Reason",
            "Reasoning Chain",
            "Matched Keywords",
        ]
    ].copy()
    accepted_margins = output.loc[output["Status"] == "Accept", "Margin"].dropna()
    if len(accepted_margins) > 0:
        low = float(np.percentile(accepted_margins, 35))
        high = float(np.percentile(accepted_margins, 65))
    else:
        low, high = None, None

    weights = None
    weights_note = "Weighted average computed using simple mean (no turnover weights found)."
    if "Turnover" in working.columns:
        w = working.loc[output["Status"] == "Accept", "Turnover"]
        if w.notna().any():
            weights = w
            weights_note = "Weighted average computed using turnover as weights."

    weighted_avg = weighted_average(accepted_margins, weights)
    safe_harbor_flag = None
    if tested_party_margin is not None and weighted_avg is not None:
        safe_harbor_flag = abs(float(tested_party_margin) - float(weighted_avg)) <= 3.0

    return output, low, high, weighted_avg, safe_harbor_flag, weights_note


def main() -> None:
    apply_theme()
    st.title("Transfer Pricing AI Reasoning Engine")
    st.markdown(
        '<div class="disclaimer">Data processed in-memory. Zero-retention policy active.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("OpenAI API Key", type="password", help="Used only for this run-time session.")
        st.subheader("Project Info")
        st.caption("Automates quantitative filters and qualitative TP comparability checks.")
        st.caption("Best used with Prowess/Capitaline Excel exports containing margin and RPT fields.")
        st.divider()
        prowess_keywords = st.text_area(
            "Search Methodology (Prowess Keywords used)",
            height=120,
            placeholder="E.g., software development services; IT enabled services; back office support; captive service provider",
        )

    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        st.subheader("Input")
        uploaded_file = st.file_uploader("Prowess/Capitaline Excel Export", type=["xlsx", "xls"])
        sheet_comp = None
        sheet_ratios = None
        tested_party_margin = st.number_input(
            "Tested Party Margin (%)",
            min_value=-100.0,
            max_value=200.0,
            value=0.0,
            step=0.1,
            help="Used for Safe Harbor Check versus weighted average of the final comparable set (±3%).",
        )
        far_analysis = st.text_area(
            "Tested Party Functional Profile (FAR Analysis)",
            height=220,
            placeholder="Describe tested party functions, assets, and risks (e.g., routine captive service provider).",
        )
        if uploaded_file:
            try:
                _, sheets = load_excel_sheets(uploaded_file)
                sheet_comp = st.selectbox("Select sheet: Comparables Data", options=sheets, index=0)
                default_ratios = 1 if len(sheets) > 1 else 0
                sheet_ratios = st.selectbox("Select sheet: Financial Ratios", options=sheets, index=default_ratios)
            except Exception as exc:
                st.error(f"Could not read Excel sheets: {exc}")
        run = st.button("Run Benchmarking Engine", type="primary")

    with right_col:
        st.subheader("Risk Dashboard")
        dashboard_placeholder = st.empty()

    if run:
        if not uploaded_file:
            st.error("Please upload an Excel file.")
            return
        if not far_analysis.strip():
            st.error("Please provide Tested Party FAR Analysis.")
            return
        if not sheet_comp or not sheet_ratios:
            st.error("Please select both the Comparables Data sheet and Financial Ratios sheet.")
            return
        if not api_key.strip():
            st.error("Please provide an OpenAI API key.")
            return

        qualitative_criteria = [
            "Functional comparability (services vs products; software service provider vs software product/IP owner).",
            "Asset profile screening (ownership of significant intangibles / proprietary products).",
            "Risk profile screening (full-fledged entrepreneur vs low-risk captive).",
            "Quantitative thresholds (RPT > 25%; persistent losses; declining turnover where available).",
            "Extraordinary events screening (merger/acquisition/demerger impacting comparability).",
        ]

        with st.spinner("Processing Rule 10B quantitative and qualitative screening..."):
            try:
                xls = pd.ExcelFile(uploaded_file)
                comparables_df = pd.read_excel(xls, sheet_name=sheet_comp)
                ratios_df = pd.read_excel(xls, sheet_name=sheet_ratios)
                merged_df = merge_comparables_and_ratios(comparables_df, ratios_df)
                client = OpenAI(api_key=api_key)
                result_df, alr_low, alr_high, wavg, safe_harbor_flag, weights_note = process_data(
                    merged_df,
                    client,
                    far_analysis,
                    tested_party_margin=float(tested_party_margin),
                )
            except Exception as exc:
                st.error(f"Processing failed: {exc}")
                return

        with right_col:
            accepted = int((result_df["Status"] == "Accept").sum())
            rejected = int((result_df["Status"] == "Reject").sum())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accepted", accepted)
            c2.metric("Rejected", rejected)
            alr_display = f"{alr_low:.2f}% - {alr_high:.2f}%" if alr_low is not None and alr_high is not None else "Not available"
            c3.metric("Final Arm's Length Range", alr_display)
            if wavg is None:
                c4.metric("Weighted Avg (final set)", "Not available")
            else:
                c4.metric("Weighted Avg (final set)", f"{wavg:.2f}%")

            if safe_harbor_flag is None:
                st.caption("Safe Harbor Check: Not available (insufficient data).")
            else:
                within = "YES" if safe_harbor_flag else "NO"
                st.markdown(f"**Safe Harbor Check (±3% vs weighted average): {within}**")
                st.caption(weights_note)

            st.markdown("### Automated Benchmarking Matrix")
            search_term = st.text_input("Search companies", value="")
            filtered_df = result_df.copy()
            if search_term.strip():
                filtered_df = filtered_df[
                    filtered_df["Company Name"].astype(str).str.contains(search_term, case=False, na=False)
                    | filtered_df["Reason"].astype(str).str.contains(search_term, case=False, na=False)
                ]
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)

            st.markdown("### Reasoning Chain (Accepted Companies)")
            accepted_rows = result_df[result_df["Status"] == "Accept"].copy()
            if accepted_rows.empty:
                st.caption("No accepted comparables to explain.")
            else:
                for _, r in accepted_rows.iterrows():
                    title = f"{r['Company Name']} — Margin: {r['Margin']:.2f}%" if pd.notna(r["Margin"]) else f"{r['Company Name']}"
                    with st.expander(title, expanded=False):
                        chain = safe_str(r.get("Reasoning Chain"))
                        kw = safe_str(r.get("Matched Keywords"))
                        st.write(chain or "Reasoning chain not provided.")
                        if kw:
                            st.caption(f"Matched keywords/phrases: {kw}")

            pdf_bytes = build_pdf(
                result_df,
                alr_low,
                alr_high,
                far_analysis=far_analysis,
                prowess_keywords=prowess_keywords,
                qualitative_criteria=qualitative_criteria,
                weighted_avg=wavg,
                safe_harbor_flag=safe_harbor_flag,
                tested_party_margin=float(tested_party_margin),
            )
            st.download_button(
                "Download Benchmarking Memorandum (Rule 10B)",
                data=pdf_bytes,
                file_name="benchmarking_memorandum_rule10b.pdf",
                mime="application/pdf",
            )

        dashboard_placeholder.empty()


if __name__ == "__main__":
    main()
