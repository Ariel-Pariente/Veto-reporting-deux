# -*- coding: utf-8 -*-
import json
import os
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from textwrap import wrap
from typing import Any, Dict, List, Optional

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


# Schémas JSON attendus (texte) pour aider le modèle en réparation
DOCUMENT_SCHEMA_TEXT = (
    '{\n'
    '  "symptomes": ["str"],\n'
    '  "examens_realises": ["str"],\n'
    '  "resultats_importants": ["str"],\n'
    '  "hypotheses_diagnostiques": [{"hypothese": "str", "arguments_pour": ["str"], "arguments_contre": ["str"], "confiance": 0.0}],\n'
    '  "traitements_proposes": ["str"],\n'
    '  "points_cles": ["str"],\n'
    '  "limitations": ["str"]\n'
    '}'
)

GLOBAL_SCHEMA_TEXT = (
    '{\n'
    '  "recurrents": [{"theme": "str", "frequence": 0, "details": ["str"]}],\n'
    '  "divergences": [{"sujet": "str", "positions": ["str"]}],\n'
    '  "infos_manquantes": ["str"],\n'
    '  "tests_recommandes": ["str"],\n'
    '  "red_flags": ["str"],\n'
    '  "recommandations": ["str"],\n'
    '  "resume_executif": "str"\n'
    '}'
)


API_KEY_FILE = Path("openai_api_key.txt")
openai_client: Optional[OpenAI] = None


def load_api_key() -> Optional[str]:
    # 1) Variable d'environnement
    key = os.environ.get("OPENAI_API_KEY")
    if key and key.strip():
        return key.strip()

    # 2) Secrets Streamlit (déploiement Streamlit Cloud)
    try:
        if "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]:
            sk = str(st.secrets["OPENAI_API_KEY"]).strip()
            if sk:
                os.environ["OPENAI_API_KEY"] = sk
                return sk
    except Exception:
        # st.secrets peut ne pas être disponible hors contexte Streamlit
        pass

    # 3) Fichier local (non commité grâce au .gitignore)
    if API_KEY_FILE.exists():
        for line in API_KEY_FILE.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            os.environ["OPENAI_API_KEY"] = s
            return s
    return None


def get_openai_client() -> OpenAI:
    global openai_client
    if openai_client is None:
        if not load_api_key():
            raise RuntimeError("Clé API OpenAI absente.")
        openai_client = OpenAI()
    return openai_client


def sanitize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    cleaned = value.replace("\ufeff", "").replace("\u2028", "\n").replace("\u2029", "\n")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    return cleaned


def extract_text_from_pdf(uploaded_file) -> str:
    uploaded_file.seek(0)
    reader = PdfReader(uploaded_file)
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            return ""
    parts: List[str] = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        txt = sanitize_text(txt)
        if txt:
            parts.append(txt)
    return "\n".join(parts).strip()


def chunk_text(s: str, max_chars: int = 4000, overlap: int = 300) -> List[str]:
    s = sanitize_text(s)
    if not s:
        return []
    if len(s) <= max_chars:
        return [s]
    chunks: List[str] = []
    start = 0
    while start < len(s):
        end = min(start + max_chars, len(s))
        chunk = s[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(s):
            break
        start = max(0, end - overlap)
    return chunks


def call_openai(messages: List[Dict[str, str]], model: str) -> str:
    client = get_openai_client()
    delay = 2
    prepared = [
        {"role": m.get("role", "user"), "content": sanitize_text(m.get("content", ""))}
        for m in messages
    ]
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=prepared,
                temperature=0.2,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            if attempt == 2:
                raise exc
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("Impossible d’obtenir une réponse du modèle.")


def parse_json_content(content: str) -> Dict[str, Any]:
    cleaned = sanitize_text(content).strip()
    if not cleaned:
        raise ValueError("Réponse LLM sans JSON valide.")
    # Retirer éventuelles fences ``` ou préfixe json
    if cleaned.startswith("```"):
        cleaned = cleaned[3:].lstrip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
        if "```" in cleaned:
            cleaned = cleaned.split("```", 1)[0]
        cleaned = cleaned.strip()
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].lstrip()
    # Candidats
    candidates: List[str] = []
    if cleaned:
        candidates.append(cleaned)
    if cleaned.startswith("{") and cleaned.endswith("}"):
        candidates.insert(0, cleaned)
    else:
        a, b = cleaned.find("{"), cleaned.rfind("}")
        if a != -1 and b != -1 and b > a:
            candidates.append(cleaned[a : b + 1])
        # Essayez d’ajouter des accolades si contenu semble type paires clé:valeur
        if "{" not in cleaned and "}" not in cleaned and '":' in cleaned:
            candidates.append("{" + cleaned + "}")
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            continue
    preview = cleaned[:160].replace("\n", " ")
    raise ValueError(f"Réponse LLM sans JSON valide: {preview}")


def ensure_json_dict(raw: str, model: str, schema_text: str) -> Dict[str, Any]:
    try:
        return parse_json_content(raw)
    except Exception:
        repair = (
            "Corrige ce JSON pour qu’il respecte exactement le schéma suivant.\n"
            f"Schéma attendu:\n{schema_text}\n\n"
            "Réponse brute:\n````\n"
            f"{sanitize_text(raw)}\n"
            "````\nNe renvoie QUE le JSON valide, sans texte autour."
        )
        messages = [
            {"role": "system", "content": "Tu corriges des réponses JSON vers un schéma validable."},
            {"role": "user", "content": repair},
        ]
        fixed = call_openai(messages, model)
        return parse_json_content(fixed)


def build_document_prompt(context: Optional[str], document_text: str) -> str:
    ctx = sanitize_text(context).strip() if context else "Aucun contexte fourni"
    return (
        "Tu es un assistant spécialisé en médecine vétérinaire. À partir du texte suivant (rapport d’interne), "
        "produis un résumé structuré JSON respectant exactement ce schéma de clés:\n"
        f"{DOCUMENT_SCHEMA_TEXT}\n"
        "Contraintes:\n- \"confiance\" est un float entre 0 et 1.\n- Pas de texte hors JSON.\n"
        "- Si une section est absente, renvoie une liste vide ou mentionne-la dans \"limitations\".\n"
        f"Contexte du cas (optionnel): {ctx}\n\n"
        "Texte du rapport:\n\n"
        f"{document_text}"
    )


def build_global_prompt(context: Optional[str], per_doc_with_names: str) -> str:
    ctx = sanitize_text(context).strip() if context else "Aucun contexte fourni"
    return (
        "Tu es un assistant qui réalise une synthèse multi-documents pour l’aide au diagnostic vétérinaire.\n"
        "À partir des résumés JSON par document, produis un JSON respectant ce schéma:\n"
        f"{GLOBAL_SCHEMA_TEXT}\n"
        "Règles:\n- \"recurrents.frequence\" = nombre de documents où le thème apparaît.\n"
        "- \"resume_executif\" = 5-8 lignes max, langage clair, actionnable.\n- Pas de texte hors JSON.\n"
        f"Contexte du cas (optionnel): {ctx}\n\n"
        "Résumés par document (JSON + nom de fichier):\n\n"
        f"{per_doc_with_names}"
    )


def summarize_document(text: str, case_context: Optional[str], model: str) -> Dict[str, Any]:
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("Texte vide après extraction.")
    partials: List[Dict[str, Any]] = []
    for chunk in chunks:
        prompt = build_document_prompt(case_context, chunk)
        messages = [
            {"role": "system", "content": "Assistant clinique structuré, précis, rigoureux."},
            {"role": "user", "content": prompt},
        ]
        raw = call_openai(messages, model)
        partials.append(ensure_json_dict(raw, model, DOCUMENT_SCHEMA_TEXT))
    if len(partials) == 1:
        return partials[0]
    # Meta-résumé
    combined = "\n\n".join(json.dumps(p, ensure_ascii=False) for p in partials)
    prompt = build_document_prompt(case_context, combined)
    messages = [
        {"role": "system", "content": "Tu intègres des résumés partiels pour produire une synthèse cohérente unique."},
        {"role": "user", "content": prompt},
    ]
    raw = call_openai(messages, model)
    return ensure_json_dict(raw, model, DOCUMENT_SCHEMA_TEXT)


def synthesize_across_docs(per_doc: List[Dict[str, Any]], case_context: Optional[str], model: str) -> Dict[str, Any]:
    payload = [
        {"filename": item["filename"], "summary": item["summary"]}
        for item in per_doc
    ]
    per_doc_json = json.dumps(payload, ensure_ascii=False, indent=2)
    prompt = build_global_prompt(case_context, per_doc_json)
    messages = [
        {"role": "system", "content": "Tu identifies convergences, divergences et recommandations actionnables."},
        {"role": "user", "content": prompt},
    ]
    raw = call_openai(messages, model)
    return ensure_json_dict(raw, model, GLOBAL_SCHEMA_TEXT)


def render_list(items: List[str]) -> None:
    if not items:
        st.markdown("- Aucun élément.")
        return
    st.markdown("\n".join(f"- {sanitize_text(x)}" for x in items if sanitize_text(x)))


def render_document_summary(doc: Dict[str, Any]) -> None:
    s = doc["summary"]
    st.markdown("**Points clés**"); render_list(s.get("points_cles", []))
    st.markdown("**Symptômes**"); render_list(s.get("symptomes", []))
    st.markdown("**Examens réalisés**"); render_list(s.get("examens_realises", []))
    st.markdown("**Résultats importants**"); render_list(s.get("resultats_importants", []))
    st.markdown("**Traitements proposés**"); render_list(s.get("traitements_proposes", []))
    st.markdown("**Limitations**"); render_list(s.get("limitations", []))
    st.markdown("**Hypothèses diagnostiques**")
    hyps = s.get("hypotheses_diagnostiques", [])
    if not hyps:
        st.markdown("- Aucune hypothèse fournie.")
    for h in hyps:
        nom = sanitize_text(h.get("hypothese", "Sans intitulé")) or "Sans intitulé"
        conf = float(h.get("confiance", 0.0))
        st.markdown(f"- **{nom}** (confiance: {conf:.2f})")
        pour = [sanitize_text(x) for x in h.get("arguments_pour", []) if sanitize_text(x)]
        contre = [sanitize_text(x) for x in h.get("arguments_contre", []) if sanitize_text(x)]
        if pour:
            st.markdown("  - Arguments pour:")
            st.markdown("\n".join(f"    - {x}" for x in pour))
        if contre:
            st.markdown("  - Arguments contre:")
            st.markdown("\n".join(f"    - {x}" for x in contre))


def build_pdf(clinic_name: str, case_context: Optional[str], per_doc: List[Dict[str, Any]], global_syn: Dict[str, Any]) -> BytesIO:
    buf = BytesIO()
    pdf = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    margin = 20 * mm
    y = height - margin

    def need(h: float) -> None:
        nonlocal y
        if y - h < margin:
            pdf.showPage()
            y = height - margin

    def head(txt: str, size: int = 14) -> None:
        nonlocal y
        need(size + 6)
        pdf.setFont("Helvetica-Bold", size)
        pdf.drawString(margin, y, txt)
        y -= size + 6

    def text(txt: str, size: int = 10, bold: bool = False) -> None:
        nonlocal y
        font = "Helvetica-Bold" if bold else "Helvetica"
        lines = wrap(txt, width=int((width - 2 * margin) / (size * 0.5)))
        for line in lines:
            need(size + 4)
            pdf.setFont(font, size)
            pdf.drawString(margin, y, line)
            y -= size + 4

    def bullet(items: List[str], size: int = 10) -> None:
        if not items:
            text("- Aucun élément.", size=size)
            return
        for it in items:
            text(f"- {it}", size=size)

    c_name = sanitize_text(clinic_name) or "Clinique"
    c_ctx = sanitize_text(case_context)

    head(f"Synthèse clinique - {c_name}", 16)
    text(f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    text(f"Nombre de documents analysés: {len(per_doc)}")
    if c_ctx:
        head("Contexte du cas", 12)
        text(c_ctx)

    for d in per_doc:
        head(f"Document: {sanitize_text(d.get('filename','Document')) or 'Document'}", 12)
        text(f"Caractères extraits: {d.get('char_count', 0)}")
        s = d.get("summary", {})
        sections = [
            ("Points clés", s.get("points_cles", [])),
            ("Symptômes", s.get("symptomes", [])),
            ("Examens réalisés", s.get("examens_realises", [])),
            ("Résultats importants", s.get("resultats_importants", [])),
            ("Traitements proposés", s.get("traitements_proposes", [])),
            ("Limitations", s.get("limitations", [])),
        ]
        for title, items in sections:
            text(title + ":", size=11, bold=True)
            bullet([sanitize_text(x) for x in items if sanitize_text(x)])
        hyps = s.get("hypotheses_diagnostiques", [])
        text("Hypothèses diagnostiques:", 11, True)
        if not hyps:
            text("- Aucune hypothèse fournie.")
        for h in hyps:
            nom = sanitize_text(h.get("hypothese", "Sans intitulé")) or "Sans intitulé"
            conf = float(h.get("confiance", 0.0))
            text(f"- {nom} (confiance: {conf:.2f})")
            text("  Arguments pour:", bold=True)
            bullet(["  " + sanitize_text(x) for x in h.get("arguments_pour", []) if sanitize_text(x)])
            text("  Arguments contre:", bold=True)
            bullet(["  " + sanitize_text(x) for x in h.get("arguments_contre", []) if sanitize_text(x)])

    head("Synthèse globale", 14)
    text("Résumé exécutif:", 11, True)
    text(sanitize_text(global_syn.get("resume_executif", "")))

    sections_g = [
        ("Points récurrents", [
            f"{sanitize_text(it.get('theme','Sans thème')) or 'Sans thème'} (fréquence: {int(it.get('frequence',0))}) - "
            + "; ".join([sanitize_text(x) for x in (it.get('details', []) or []) if sanitize_text(x)])
            for it in (global_syn.get("recurrents", []) or [])
        ]),
        ("Divergences", [
            f"{sanitize_text(it.get('sujet','Sans sujet')) or 'Sans sujet'}: "
            + "; ".join([sanitize_text(x) for x in (it.get('positions', []) or []) if sanitize_text(x)])
            for it in (global_syn.get("divergences", []) or [])
        ]),
        ("Informations manquantes", global_syn.get("infos_manquantes", [])),
        ("Tests recommandés", global_syn.get("tests_recommandes", [])),
        ("Red flags", global_syn.get("red_flags", [])),
        ("Recommandations", global_syn.get("recommandations", [])),
    ]
    for title, items in sections_g:
        text(title + ":", 11, True)
        bullet([sanitize_text(x) for x in items if sanitize_text(x)])

    text("Outil d’aide à la décision — ne remplace pas le jugement clinique.", 9, True)
    pdf.showPage()
    pdf.save()
    buf.seek(0)
    return buf


# ------------------------- UI -------------------------
st.set_page_config(page_title="Synthèse de rapports vétérinaires (MVP)", layout="wide")
st.title("Synthèse de rapports vétérinaires (MVP)")

clinic_name = st.text_input("Nom de la clinique", value="Frégis")
case_context = st.text_area("Contexte du cas (optionnel)")
model_choice = st.selectbox("Modèle OpenAI", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"])
uploaded_files = st.file_uploader("Déposez plusieurs PDF", type=["pdf"], accept_multiple_files=True)
st.code("streamlit run app.py", language="bash")

api_key = load_api_key()
if not api_key:
    st.warning("Clé API OpenAI absente. Ajoutez-la dans openai_api_key.txt (une ligne) ou définissez OPENAI_API_KEY.")

if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None

if st.button("Analyser"):
    if not uploaded_files:
        st.warning("Veuillez déposer au moins un PDF.")
    elif not load_api_key():
        st.error("Analyse impossible sans clé API OpenAI.")
    else:
        results: List[Dict[str, Any]] = []
        with st.spinner("Analyse en cours..."):
            for up in uploaded_files:
                txt = extract_text_from_pdf(up)
                if not txt:
                    st.warning(f"{up.name}: OCR non pris en charge ou texte vide.")
                    continue
                try:
                    summ = summarize_document(txt, case_context, model_choice)
                    results.append({"filename": up.name, "char_count": len(txt), "summary": summ})
                except Exception as exc:
                    st.error(f"Erreur lors du résumé de {up.name}: {exc}")
            global_syn = None
            if results:
                try:
                    global_syn = synthesize_across_docs(results, case_context, model_choice)
                except Exception as exc:
                    st.error(f"Erreur lors de la synthèse globale: {exc}")
            if results and global_syn:
                st.session_state["analysis_result"] = {
                    "clinic_name": clinic_name,
                    "case_context": case_context,
                    "per_doc": results,
                    "global_syn": global_syn,
                }
            else:
                st.session_state["analysis_result"] = None

analysis = st.session_state.get("analysis_result")
if analysis:
    st.subheader("Fichiers analysés")
    st.table([
        {"Fichier": d["filename"], "Caractères extraits": d["char_count"]} for d in analysis["per_doc"]
    ])

    for d in analysis["per_doc"]:
        with st.expander(f"Résumé — {d['filename']}"):
            render_document_summary(d)

    gs = analysis["global_syn"]
    st.subheader("Synthèse globale")
    st.markdown("**Résumé exécutif**")
    st.write(sanitize_text(gs.get("resume_executif", "")))

    if gs.get("recurrents"):
        st.markdown("**Points récurrents**")
        for it in gs["recurrents"]:
            details = "\n".join(f"  - {sanitize_text(x)}" for x in it.get("details", []) if sanitize_text(x))
            st.markdown(f"- **{sanitize_text(it.get('theme','Sans thème')) or 'Sans thème'}** (fréquence: {int(it.get('frequence',0))})\n{details}")
    if gs.get("divergences"):
        st.markdown("**Divergences**")
        for it in gs["divergences"]:
            positions = "\n".join(f"  - {sanitize_text(x)}" for x in it.get("positions", []) if sanitize_text(x))
            st.markdown(f"- **{sanitize_text(it.get('sujet','Sans sujet')) or 'Sans sujet'}**\n{positions}")
    if gs.get("infos_manquantes"):
        st.markdown("**Informations manquantes**"); render_list(gs["infos_manquantes"])
    if gs.get("tests_recommandes"):
        st.markdown("**Tests recommandés**"); render_list(gs["tests_recommandes"])
    if gs.get("red_flags"):
        st.markdown("**Red flags**"); render_list(gs["red_flags"])
    if gs.get("recommandations"):
        st.markdown("**Recommandations**"); render_list(gs["recommandations"])

    pdf_buffer = build_pdf(analysis["clinic_name"], analysis["case_context"], analysis["per_doc"], analysis["global_syn"])
    st.download_button(
        "Télécharger la synthèse en PDF",
        data=pdf_buffer,
        file_name="synthese_veterinaire.pdf",
        mime="application/pdf",
    )

st.caption("Outil d’aide à la décision — ne remplace pas le jugement clinique.")
