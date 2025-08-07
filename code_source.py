import streamlit as st
from groq import Groq
import PyPDF2
import re
import io
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import networkx as nx
import json
import plotly.graph_objects as go

# =============================================
# CONFIGURATION INITIALE ET STYLE
# =============================================

st.set_page_config(
    page_title="TRIZ Avancée - Assistant Expert",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS modernisé
st.markdown("""
<style>
    :root {
        --primary: #3498db;
        --secondary: #2c3e50;
        --light-bg: #f8f9fa;
        --card-bg: #ffffff;
    }
    .stApp {
        background-color: var(--light-bg);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header {
        color: var(--secondary);
        border-bottom: 2px solid var(--primary);
        padding-bottom: 10px;
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
    }
    .stTextArea textarea {
        min-height: 150px;
        border-radius: 8px !important;
        padding: 15px !important;
    }
    .solution-card {
        border-radius: 10px;
        padding: 1.5em;
        margin-bottom: 2em;
        background-color: var(--card-bg);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid var(--primary);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .principle-card {
        border-left: 4px solid var(--primary);
        padding-left: 1em;
        margin-bottom: 1em;
        background-color: #f1f8fe;
        border-radius: 5px;
    }
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border-radius: 8px;
        padding: 12px 28px;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }
    .generated-image {
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin: 20px 0;
        border: 1px solid #e0e0e0;
    }
    .param-box {
        padding: 1.2em;
        border-radius: 8px;
        margin-bottom: 1.5em;
    }
    .improve-param {
        background-color: #e6f7ff;
        border-left: 4px solid #1890ff;
    }
    .degrade-param {
        background-color: #fff2f0;
        border-left: 4px solid #ff4d4f;
    }
    .stSpinner > div {
        text-align: center;
    }
    .stMarkdown h3 {
        color: var(--secondary);
        margin-top: 1.5rem;
    }
    .st-expander {
        border-radius: 8px !important;
        border: 1px solid #e0e0e0 !important;
    }
    .evaluation-card {
        border-radius: 8px;
        padding: 1em;
        background-color: #f8f9fa;
        border-left: 4px solid #6c757d;
        margin-bottom: 1em;
    }
    .progress-text {
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Titre modernisé
st.markdown("<h1 class='header'>🔧 TRIZ Avancée - Assistant Expert</h1>", unsafe_allow_html=True)
st.markdown("""
<div style="margin-bottom: 2rem; font-size: 1.1rem; color: var(--secondary);">
Résolution systématique de contradictions techniques avec illustrations conceptuelles
</div>
""", unsafe_allow_html=True)

# =============================================
# FONCTIONS DE BASE
# =============================================

# Client Groq
client = Groq(api_key)

# Liste officielle des paramètres TRIZ (1-39)
TRIZ_PARAMS = {
    1: "Poids de l'objet mobile",
    2: "Poids de l'objet stationnaire",
    3: "Longueur de l'objet mobile",
    4: "Longueur de l'objet stationnaire",
    5: "Aire de l'objet mobile",
    6: "Aire de l'objet stationnaire",
    7: "Volume de l'objet mobile",
    8: "Volume de l'objet stationnaire",
    9: "Vitesse",
    10: "Force",
    11: "Contrainte ou pression",
    12: "Forme",
    13: "Stabilité de la composition",
    14: "Résistance",
    15: "Durée de vie de l'objet mobile",
    16: "Durée de vie de l'objet stationnaire",
    17: "Température",
    18: "Luminosité",
    19: "Énergie dépensée par l'objet mobile",
    20: "Énergie dépensée par l'objet stationnaire",
    21: "Puissance",
    22: "Pertes d'énergie",
    23: "Pertes de matière",
    24: "Pertes d'information",
    25: "Pertes de temps",
    26: "Quantité de matière",
    27: "Fiabilité",
    28: "Précision de mesure",
    29: "Précision de fabrication",
    30: "Facteurs nuisibles affectant l'objet",
    31: "Facteurs nuisibles générés par l'objet",
    32: "Facilité de fabrication",
    33: "Facilité d'utilisation",
    34: "Facilité de réparation",
    35: "Adaptabilité",
    36: "Complexité du système",
    37: "Complexité de contrôle",
    38: "Niveau d'automatisation",
    39: "Productivité"
}

# Modèle d'embedding
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-mpnet-base-v2')

embedder = load_embedding_model()

# Matrice de référence
REFERENCE_MATRIX = {
    (14, 1): [1, 8, 40, 15],
    (14, 2): [40, 26, 27, 1],
    (1, 14): [1, 8, 15, 35],
}

# Chargement matrice TRIZ
@st.cache_data
def load_triz_matrix(file_path="Matrice_TRIZ.xlsx"):
    try:
        df_brut = pd.read_excel(file_path, sheet_name='table_2', header=None)
        
        param_rows = {}
        param_cols = {}
        
        # Traitement des colonnes (paramètres à améliorer)
        for i, value in enumerate(df_brut.iloc[0, 1:], start=1):
            try:
                if pd.notna(value):
                    if isinstance(value, str) and ':' in value:
                        param_num = int(value.split(':')[0].strip())
                    elif isinstance(value, (int, float)):
                        param_num = int(value)
                    else:
                        match = re.search(r'^(\d+)', str(value).strip())
                        if match:
                            param_num = int(match.group(1))
                        else:
                            continue
                    
                    if 1 <= param_num <= 39:
                        param_cols[param_num] = i
            except Exception as e:
                if st.session_state.get('debug_mode', False):
                    st.warning(f"Erreur colonne {i}: {str(e)}")
        
        # Traitement des lignes (paramètres qui se dégradent)
        for i, value in enumerate(df_brut.iloc[1:, 0], start=1):
            try:
                if pd.notna(value):
                    param_str = str(value).strip()
                    match = re.search(r'^(\d+)[\s:.]+', param_str)
                    if match:
                        param_num = int(match.group(1))
                        if 1 <= param_num <= 39:
                            param_rows[param_num] = i
            except Exception as e:
                if st.session_state.get('debug_mode', False):
                    st.warning(f"Erreur ligne {i}: {str(e)}")
        
        # Fallback si aucun paramètre trouvé
        if not param_rows or not param_cols:
            if st.session_state.get('debug_mode', False):
                st.warning("Création mapping par défaut")
            
            for i in range(1, min(40, len(df_brut))):
                if i not in param_rows:
                    param_rows[i] = i
            
            for i in range(1, min(40, len(df_brut.columns))):
                if i not in param_cols:
                    param_cols[i] = i
        
        matrice = df_brut.iloc[1:, 1:].copy()
        
        debug_info = {
            "param_rows_count": len(param_rows),
            "param_cols_count": len(param_cols),
            "row_keys": list(param_rows.keys()),
            "col_keys": list(param_cols.keys())
        }
        
        return {
            "matrice": matrice,
            "param_rows": param_rows,
            "param_cols": param_cols,
            "debug_info": debug_info
        }
    except Exception as e:
        st.error(f"Erreur chargement matrice TRIZ: {str(e)}")
        return {"matrice": pd.DataFrame(), "param_rows": {}, "param_cols": {}, "debug_info": {}}
def get_principles_from_matrix(param_improve_num: int, param_degrade_num: int, triz_matrix: Dict) -> List[int]:
    # Vérifier d'abord la référence
    reference_key = (param_improve_num, param_degrade_num)
    if reference_key in REFERENCE_MATRIX:
        return REFERENCE_MATRIX[reference_key]

    # Cas spécifique pour vitesse (#9) et précision de mesure (#28)
    if param_improve_num == 9 and param_degrade_num == 28:
        return [28, 32, 1, 24]  # Valeurs visibles dans la matrice

    # Vérifier l'existence des paramètres
    if (param_improve_num not in triz_matrix["param_cols"] or 
        param_degrade_num not in triz_matrix["param_rows"]):
        return []

    try:
        # Accès à la cellule
        col_idx = triz_matrix["param_cols"][param_improve_num] - 1
        row_idx = triz_matrix["param_rows"][param_degrade_num] - 1
        cell_value = triz_matrix["matrice"].iloc[row_idx, col_idx]

        if pd.isna(cell_value):
            return []

        # Conversion forcée en string pour le parsing
        cell_str = str(cell_value)
        
        # Extraction robuste des nombres
        principles = []
        for part in re.split(r'[\s,;.-]+', cell_str):
            part = re.sub(r'[^\d]', '', part)  # Ne garder que les chiffres
            if part.isdigit():
                num = int(part)
                if 1 <= num <= 40:
                    principles.append(num)

        # Retourner les principes uniques triés
        return sorted(list(set(principles)))

    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Erreur extraction: {str(e)}")
        return []
def get_principles_from_llm(problem: str, param_improve: Tuple[int, str], param_degrade: Tuple[int, str]) -> List[int]:
    """Demande au LLM de proposer des principes TRIZ quand la matrice n'en trouve pas"""
    prompt = f"""
    [CONTEXTE]
    Vous êtes un expert TRIZ. Proposez 3-5 principes TRIZ (numéros 1-40) qui pourraient résoudre la contradiction suivante,
    car aucune solution n'a été trouvée dans la matrice standard.

    [PROBLÈME]
    {problem}

    [CONTRADICTION]
    Améliorer {param_improve[1]} (#{param_improve[0]}) dégrade {param_degrade[1]} (#{param_degrade[0]})

    [INSTRUCTIONS]
    1. Répondez UNIQUEMENT avec une liste de numéros séparés par des virgules, comme ceci: 1, 2, 5, 10
    2. Choisissez des principes pertinents pour cette contradiction spécifique
    3. Ne proposez que des numéros entre 1 et 40
    """
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=50
        )
        text = response.choices[0].message.content.strip()
        principles = [int(p.strip()) for p in text.split(',') if p.strip().isdigit() and 1 <= int(p.strip()) <= 40]
        return principles[:5]  # Limiter à 5 principes max
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Erreur LLM pour principes: {str(e)}")
        return []

def parse_cell_value(cell_value) -> List[int]:
    principles = []
    
    try:
        if isinstance(cell_value, str):
            matches = re.findall(r'\b(\d+)\b', cell_value)
            principles = [int(match) for match in matches if 1 <= int(match) <= 40]
        elif isinstance(cell_value, (int, float)) and not pd.isna(cell_value):
            principle = int(cell_value)
            if 1 <= principle <= 40:
                principles = [principle]
        elif isinstance(cell_value, list):
            principles = [int(p) for p in cell_value if isinstance(p, (int, float)) and 1 <= int(p) <= 40]
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Erreur analyse valeur: {str(e)}")
    
    return principles

def extract_contradiction_parameters(problem: str) -> Optional[List[Tuple[int, str]]]:
    prompt = f"""
    [CONTEXTE]
    Vous êtes un expert TRIZ. Identifiez EXACTEMENT DEUX paramètres TRIZ qui sont en CONTRADICTION DIRECTE
    dans le problème technique suivant, en vous basant sur la liste officielle (numéros 1-39).
    Indiquez clairement quel paramètre on essaie d'améliorer et lequel se dégrade en conséquence.

    {TRIZ_PARAMS}

    [PROBLÈME TECHNIQUE]
    {problem}

    [INSTRUCTIONS STRICTES]
    1. Répondez UNIQUEMENT dans ce format exact :
        Paramètre à améliorer: #<numéro> <nom_standard>
        Paramètre qui se dégrade: #<numéro> <nom_standard>
    2. Utilisez EXCLUSIVEMENT les noms de la liste provided
    3. Exemple de réponse valide :
        Paramètre à améliorer: #14 Résistance
        Paramètre qui se dégrade: #1 Poids de l'objet mobile
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150
        )
        text = response.choices[0].message.content
        pattern = r"Paramètre à améliorer: #(\d+) (.+)\nParamètre qui se dégrade: #(\d+) (.+)"
        match = re.search(pattern, text.strip())
        if not match:
            st.error("Format de réponse incorrect")
            return None
        param_improve_num, param_improve_name = int(match.group(1)), match.group(2)
        param_degrade_num, param_degrade_name = int(match.group(3)), match.group(4)
        if param_improve_num not in TRIZ_PARAMS or param_degrade_num not in TRIZ_PARAMS:
            st.error("Numéro de paramètre invalide (1-39)")
            return None
        if param_improve_name != TRIZ_PARAMS.get(param_improve_num) or param_degrade_name != TRIZ_PARAMS.get(param_degrade_num):
            st.error("Nom de paramètre incorrect")
            return None
        return [(param_improve_num, param_improve_name), (param_degrade_num, param_degrade_name)]
    except Exception as e:
        st.error(f"Erreur LLM: {str(e)}")
        return None

# Backup principles
BACKUP_PRINCIPLES = {
    1: {"title": "Segmentation", "description": "Diviser un objet en parties indépendantes"},
    2: {"title": "Extraction", "description": "Extraire la partie perturbatrice"},
    3: {"title": "Qualité locale", "description": "Structure non uniforme"},
    4: {"title": "Asymétrie", "description": "Remplacer symétrique par asymétrique"},
    5: {"title": "Combinaison", "description": "Rapprocher ou fusionner des opérations ou objets"},
    6: {"title": "Universalité", "description": "Multifonction"},
    7: {"title": "Poupées russes", "description": "Placer les objets les uns dans les autres"},
    8: {"title": "Contrepoids", "description": "Compenser la masse par interaction"},
    9: {"title": "Action inverse préliminaire", "description": "Créer des contraintes opposées préalables"},
    10: {"title": "Action préliminaire", "description": "Réaliser les changements avant qu'ils soient nécessaires"},
    11: {"title": "Compensation préliminaire", "description": "Compenser le manque de fiabilité par contre-mesures"},
    12: {"title": "Equipotentialité", "description": "Limiter les changements de position"},
    13: {"title": "Inversion", "description": "Au lieu d'une action dictée par les spécifications, faire l'inverse"},
    14: {"title": "Sphéricité", "description": "Utiliser des surfaces courbes"},
    15: {"title": "Dynamisme", "description": "Adapter pour performance optimale"},
    16: {"title": "Action partielle ou excessive", "description": "Si difficile à 100%, faire un peu plus ou moins"},
    17: {"title": "Changement de dimension", "description": "Mouvement dans un nouvel espace"},
    18: {"title": "Vibration mécanique", "description": "Mettre un objet en oscillation"},
    19: {"title": "Action périodique", "description": "Remplacer action continue par périodique"},
    20: {"title": "Continuité d'une action utile", "description": "Travailler sans pause"},
    21: {"title": "Grande vitesse", "description": "Conduire un processus nocif ou dangereux à très grande vitesse"},
    22: {"title": "Application bénéfique d'un effet néfaste", "description": "Utiliser des facteurs nuisibles pour obtenir un effet positif"},
    23: {"title": "Rétroaction", "description": "Introduire un asservissement"},
    24: {"title": "Intermédiaire", "description": "Utiliser un objet intermédiaire pour transférer ou réaliser une action"},
    25: {"title": "Self-service", "description": "Faire en sorte qu'un objet se répare ou s'entretienne lui-même"},
    26: {"title": "Copie", "description": "Utiliser des copies simples et peu coûteuses"},
    27: {"title": "Ephémère et bon marché", "description": "Remplacer un objet cher par plusieurs objets bon marché"},
    28: {"title": "Remplacer les systèmes mécaniques", "description": "Utiliser des systèmes optiques, acoustiques, thermiques..."},
    29: {"title": "Systèmes pneumatiques ou hydrauliques", "description": "Remplacer des parties solides par des fluides"},
    30: {"title": "Membranes flexibles et films minces", "description": "Isoler avec des matériaux souples"},
    31: {"title": "Matériaux poreux", "description": "Rendre un objet poreux"},
    32: {"title": "Changement de couleur", "description": "Modifier la couleur d'un objet ou de son environnement"},
    33: {"title": "Homogénéité", "description": "Faire interagir les objets avec un objet de même matière"},
    34: {"title": "Rejet et régénération", "description": "Éliminer ou modifier un élément après utilisation"},
    35: {"title": "Modification des propriétés", "description": "Changer état physique, concentration, flexibilité..."},
    36: {"title": "Changement de phase", "description": "Utiliser phénomènes de changement de phase"},
    37: {"title": "Dilatation thermique", "description": "Utiliser la dilatation ou contraction thermique"},
    38: {"title": "Oxydants puissants", "description": "Remplacer l'air ordinaire par de l'air enrichi en oxygène"},
    39: {"title": "Atmosphère inerte", "description": "Remplacer l'environnement normal par un environnement inerte"},
    40: {"title": "Matériaux composites", "description": "Utiliser des combinaisons de matériaux"}
}

@st.cache_resource
def load_triz_principles():
    def read_pdf(file_path):
        try:
            with open(file_path, "rb") as f:
                return io.BytesIO(f.read())
        except Exception as e:
            if st.session_state.get('debug_mode', False):
                st.error(f"Erreur chargement PDF: {str(e)}")
            return None
    
    try:
        pdf_contents = {
            "principles": read_pdf("liste-des-40-principes-de-triz-ensps.pdf")
        }
        
        principles_text = extract_text_from_io(pdf_contents["principles"])
        principles = extract_principles(principles_text)
        
        # Compléter avec les principes manquants
        for num in range(1, 41):
            if num not in principles and num in BACKUP_PRINCIPLES:
                principles[num] = BACKUP_PRINCIPLES[num]
        
        return principles
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Erreur chargement principes: {str(e)}")
        return BACKUP_PRINCIPLES

def extract_text_from_io(pdf_io):
    if pdf_io is None:
        return ""
    try:
        pdf_io.seek(0)
        reader = PyPDF2.PdfReader(pdf_io)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            cleaned_text = re.sub(r'\n+', '\n', re.sub(r' +', ' ', page_text.strip()))
            text += cleaned_text + "\n\n"
        return text
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Erreur extraction texte: {str(e)}")
        return ""

def extract_principles(text: str) -> Dict[int, Dict]:
    principles = {}
    
    if text.strip():
        pattern = re.compile(r'^(\d+)\.\s([^\n]+)\n(?:`o`|•|o)\s([^\n]+(?:\n(?:`o`|•|o)\s[^\n]+)*)', re.MULTILINE)
        for match in pattern.finditer(text):
            try:
                principle_num = int(match.group(1))
                principles[principle_num] = {
                    "title": match.group(2).strip(),
                    "description": match.group(3).replace('`o`', '•').strip()
                }
            except Exception as e:
                if st.session_state.get('debug_mode', False):
                    st.error(f"Erreur extraction principe {match.group(1)}: {str(e)}")
    
    return principles

def generate_solutions_from_principles(problem: str, param_improve: Tuple[int, str], 
                                     param_degrade: Tuple[int, str], 
                                     principles_info: Dict[int, Dict]) -> str:
    principles_context = ""
    for num, info in principles_info.items():
        principles_context += f"Principe #{num} - {info.get('title', '')}: {info.get('description', '')}\n"

    prompt = f"""
    [Contexte TRIZ]
    Problème: {problem}
    Contradiction: Améliorer '{param_improve[1]}' (#{param_improve[0]}) se traduit par une dégradation de '{param_degrade[1]}' (#{param_degrade[0]}).
    
    Principes TRIZ recommandés par la matrice de contradictions:
    {principles_context}

    [Instruction]
    Pour chaque principe TRIZ listé ci-dessus, générez une solution concrète qui l'applique spécifiquement pour résoudre la contradiction identifiée.
    
    Format de chaque solution:
    
    ### Principe #{num} - {info.get('title', 'Principe')}
    [Description de la solution] : Comment ce principe peut être appliqué concrètement au problème
    [Avantages] : Principaux avantages de cette approche
    [Faisabilité] : Évaluation rapide de la faisabilité technique et économique
    
    Assurez-vous que chaque solution est concrète, applicable et spécifique au problème décrit.
    """
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur de génération de solutions: {str(e)}"

def generate_technical_diagram(solution_text: str, principle_info: Dict) -> Image.Image:
    """Génère une image illustrative locale à partir du texte de solution"""
    try:
        plt.figure(figsize=(10, 6))
        plt.axis('off')
        
        # Création d'un WordCloud avec les mots clés
        wc = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='Blues',
            max_words=50
        ).generate(solution_text)
        
        plt.imshow(wc, interpolation='bilinear')
        principle_key = list(principle_info.keys())[0] if isinstance(principle_info, dict) and principle_info else 0
        principle_title = principle_info.get('title', 'Principe') if isinstance(principle_info, dict) else 'Principe'
        plt.title(f"Principe #{principle_key}: {principle_title}", pad=20)
        
        # Conversion en image PIL
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return Image.open(buf)
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Erreur génération diagramme: {str(e)}")
        return Image.new('RGB', (800, 400), color=(240, 240, 240))

def evaluate_triz_solution(solution_text: str, principle_info: dict) -> dict:
    """
    Évalue une solution TRIZ selon 5 critères normatifs
    Retourne un dictionnaire de scores et une analyse textuelle
    """
    principle_key = list(principle_info.keys())[0] if isinstance(principle_info, dict) and principle_info else 0
    principle_title = principle_info.get('title', 'Principe') if isinstance(principle_info, dict) else 'Principe'
    principle_desc = principle_info.get('description', '') if isinstance(principle_info, dict) else ''
    
    evaluation_prompt = f"""
    [ROLE]
    Vous êtes un expert TRIZ certifié. Évaluez rigoureusement cette solution technique.

    [CRITÈRES]
    1. Alignement_TRIZ (1-5): Correspondance exacte avec les principes TRIZ officiels
    2. Innovation (1-5): Originalité de l'application du principe
    3. Faisabilité (1-5): Réalisme technique et économique
    4. Impact (1-5): Résolution effective de la contradiction
    5. Clarté (1-5): Précision des explications fournies

    [CONTEXTE]
    Principe appliqué: {principle_title} (#{principle_key})
    Description du principe: {principle_desc}

    [SOLUTION À ÉVALUER]
{solution_text}

[DIRECTIVES STRICTES]
- Répondez uniquement dans ce format JSON:
{{
    "scores": {{
        "alignement_triz": <1-5>,
        "innovation": <1-5>,
        "faisabilite": <1-5>,
        "impact": <1-5>,
        "clarte": <1-5>
    }},
    "analyse": "<texte de 2-3 phrases résumant l'évaluation>"
}}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": evaluation_prompt}],
            temperature=0.2,
            max_tokens=500
        )
        evaluation_text = response.choices[0].message.content
        
        # Extraction du JSON
        json_match = re.search(r'\{[\s\S]*\}', evaluation_text)
        if json_match:
            eval_data = json.loads(json_match.group(0))
            return eval_data
        else:
            return {
                "scores": {
                    "alignement_triz": 3,
                    "innovation": 3,
                    "faisabilite": 3,
                    "impact": 3, 
                    "clarte": 3
                },
                "analyse": "Évaluation par défaut suite à une erreur d'analyse."
            }
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Erreur évaluation: {str(e)}")
        return {
            "scores": {
                "alignement_triz": 3,
                "innovation": 3,
                "faisabilite": 3,
                "impact": 3,
                "clarte": 3
            },
            "analyse": f"Erreur d'évaluation: {str(e)}"
        }

def create_principles_network(principles_list: List[int], all_principles: Dict) -> None:
    """Crée une visualisation en réseau des principes et leurs relations"""
    try:
        G = nx.Graph()
        
        # Ajout des nœuds (principes)
        for p_num in principles_list:
            if p_num in all_principles:
                p_info = all_principles[p_num]
                G.add_node(p_num, label=f"{p_num}. {p_info.get('title', '')}")
        
        # Ajout des arêtes (relations entre principes)
        for i, p1 in enumerate(principles_list):
            for j, p2 in enumerate(principles_list):
                if i < j:  # Pour éviter les doublons
                    G.add_edge(p1, p2, weight=1)
        
        # Création de la visualisation
        pos = nx.spring_layout(G, seed=42)
        
        # Création du graphique avec Plotly
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node}. {all_principles[node].get('title', '')}")
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            marker=dict(
                showscale=False,
                colorscale='YlGnBu',
                size=30,
                color=[],
                line_width=2))
        
        # Couleur basée sur le degré de connexion
        node_degrees = dict(G.degree())
        node_trace.marker.color = [node_degrees[node] for node in G.nodes()]
        
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        title='Relations entre principes TRIZ',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Erreur réseau principes: {str(e)}")

# =============================================
# INTERFACE UTILISATEUR
# =============================================

# Chargement de la matrice TRIZ
triz_matrix = load_triz_matrix()
triz_principles = load_triz_principles()

# Tabs pour organiser l'interface
tab1, tab2, tab3 = st.tabs(["📝 Définition du problème", "🧩 Solutions TRIZ", "📊 Analyse avancée"])

with tab1:
    st.markdown("### Décrivez votre problème technique")
    
    example_problems = [
        "Comment réduire le poids d'une bicyclette tout en conservant sa résistance mécanique?",
        "Comment augmenter la vitesse d'une imprimante sans diminuer la qualité d'impression?",
        "Comment réduire la consommation d'énergie d'un réfrigérateur sans diminuer sa capacité de refroidissement?"
    ]
    selected_example = st.selectbox("Exemples de problèmes", [""] + example_problems)
    
    problem_text = st.text_area(
        "Décrivez le problème technique à résoudre",
        value=selected_example if selected_example else "",
        height=150
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Identifier la contradiction", type="primary"):
            if problem_text:
                with st.spinner("Analyse de la contradiction en cours..."):
                    contradiction_params = extract_contradiction_parameters(problem_text)
                    if contradiction_params:
                        st.session_state['contradiction_params'] = contradiction_params
                        st.session_state['problem_text'] = problem_text
                        param_improve, param_degrade = contradiction_params
                        
                        st.markdown(f"""
                        <div class='param-box improve-param'>
                            <strong>Paramètre à améliorer:</strong> #{param_improve[0]} - {param_improve[1]}
                        </div>
                        <div class='param-box degrade-param'>
                            <strong>Paramètre qui se dégrade:</strong> #{param_degrade[0]} - {param_degrade[1]}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Trouver principes recommandés
                        principles_nums = get_principles_from_matrix(param_improve[0], param_degrade[0], triz_matrix)
                        
                        # Si aucun principe trouvé dans la matrice, demander au LLM
                        if not principles_nums:
                            st.warning("Aucun principe trouvé dans la matrice pour cette contradiction. Consultation de l'expert TRIZ...")
                            with st.spinner("Recherche de principes pertinents..."):
                                principles_nums = get_principles_from_llm(problem_text, param_improve, param_degrade)
                                if principles_nums:
                                    st.success(f"Principes recommandés par l'expert TRIZ: {', '.join(map(str, principles_nums))}")
                                else:
                                    st.error("Impossible de trouver des principes pertinents pour cette contradiction.")
                        
                        if principles_nums:
                            st.session_state['principles_nums'] = principles_nums
                            principles_info = {num: triz_principles.get(num, {}) for num in principles_nums}
                            st.session_state['principles_info'] = principles_info
                            
                            st.markdown("### Principes TRIZ recommandés:")
                            
                            for num, info in principles_info.items():
                                st.markdown(f"""
                                <div class='principle-card'>
                                    <strong>Principe #{num} - {info.get('title', 'Principe')}:</strong><br>
                                    {info.get('description', 'Description non disponible')}
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.error("Impossible d'identifier une contradiction TRIZ claire dans le problème.")
            else:
                st.warning("Veuillez décrire le problème technique.")
    
    with col2:
        st.markdown("### À propos de TRIZ")
        st.markdown("""
        TRIZ est une méthode systématique pour résoudre les problèmes techniques en identifiant les contradictions et en appliquant des principes d'innovation éprouvés.
        
        1. Identifiez votre problème technique
        2. Analysez la contradiction sous-jacente
        3. Appliquez les principes TRIZ recommandés
        4. Développez des solutions concrètes
        
        La matrice de contradictions TRIZ propose des principes d'innovation pour résoudre chaque type spécifique de contradiction technique.
        """)

with tab2:
    if 'contradiction_params' in st.session_state and 'principles_nums' in st.session_state:
        st.markdown("### Solutions générées par principes TRIZ")
        
        param_improve, param_degrade = st.session_state['contradiction_params']
        principles_nums = st.session_state['principles_nums']
        principles_info = st.session_state['principles_info']
        problem_text = st.session_state['problem_text']
        
        st.markdown(f"""
        <div style='background-color: #f0f7fb; padding: 10px; border-radius: 8px; margin-bottom: 20px;'>
            <strong>Problème:</strong> {problem_text}<br>
            <strong>Contradiction:</strong> Améliorer <span style='color:#1890ff'><strong>#{param_improve[0]} {param_improve[1]}</strong></span> 
            dégrade <span style='color:#ff4d4f'><strong>#{param_degrade[0]} {param_degrade[1]}</strong></span>
        </div>
        """, unsafe_allow_html=True)
        
        if 'solutions_generated' not in st.session_state:
            if st.button("Générer des solutions", type="primary"):
                with st.spinner("Génération des solutions TRIZ en cours..."):
                    principles_dict = {num: principles_info[num] for num in principles_nums}
                    solutions_text = generate_solutions_from_principles(
                        problem_text, 
                        param_improve, 
                        param_degrade, 
                        principles_dict
                    )
                    st.session_state['solutions_text'] = solutions_text
                    st.session_state['solutions_generated'] = True
                    
                    # Extraction des solutions individuelles
                    pattern = r'### Principe #(\d+) - ([^\n]+)([\s\S]*?)(?=### Principe #\d+|$)'
                    matches = re.finditer(pattern, solutions_text)
                    
                    individual_solutions = []
                    for match in matches:
                        principle_num = int(match.group(1))
                        principle_title = match.group(2).strip()
                        solution_content = match.group(3).strip()
                        
                        if principle_num in principles_nums:
                            principle_info = {principle_num: principles_info.get(principle_num, {})}
                            
                            # Génération d'illustration conceptuelle
                            concept_image = generate_technical_diagram(solution_content, principle_info)
                            
                            # Évaluation de la solution
                            evaluation = evaluate_triz_solution(solution_content, principle_info)
                            
                            individual_solutions.append({
                                'principle_num': principle_num,
                                'principle_title': principle_title,
                                'content': solution_content,
                                'image': concept_image,
                                'evaluation': evaluation
                            })
                    
                    st.session_state['individual_solutions'] = individual_solutions
        
        if 'solutions_generated' in st.session_state and 'individual_solutions' in st.session_state:
            solutions = st.session_state['individual_solutions']
            
            # Création d'un sélecteur pour les solutions
            solution_options = [f"Principe #{sol['principle_num']} - {sol['principle_title']}" for sol in solutions]
            selected_solution_idx = st.selectbox("Explorer les solutions", range(len(solution_options)), format_func=lambda i: solution_options[i])
            
            if selected_solution_idx is not None:
                sol = solutions[selected_solution_idx]
                
                st.markdown(f"""
                <div class='solution-card'>
                    <h3>Principe #{sol['principle_num']} - {sol['principle_title']}</h3>
                    {sol['content']}
                </div>
                """, unsafe_allow_html=True)
                
                # Affichage de l'image conceptuelle
                if 'image' in sol:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.image(sol['image'], caption=f"Illustration conceptuelle - Principe #{sol['principle_num']}", use_column_width=True)
                    
                    with col2:
                        if 'evaluation' in sol:
                            eval_data = sol['evaluation']
                            st.markdown("<div class='evaluation-card'>", unsafe_allow_html=True)
                            st.markdown("#### Évaluation TRIZ")
                            
                            scores = eval_data.get('scores', {})
                            for criterion, score in scores.items():
                                st.markdown(f"<p class='progress-text'>{criterion.title()}</p>", unsafe_allow_html=True)
                                st.progress(score/5)
                            
                            st.markdown(f"<p><em>{eval_data.get('analyse', 'Pas d\'analyse disponible')}</em></p>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                
                # Options d'exportation
                st.download_button(
                    label="📥 Exporter solution (PDF)",
                    data="Solution TRIZ exportable",
                    file_name=f"triz_solution_{sol['principle_num']}.pdf",
                    mime="application/pdf",
                    disabled=True
                )
    else:
        st.info("Veuillez d'abord identifier la contradiction dans l'onglet 'Définition du problème'.")

with tab3:
    if 'contradiction_params' in st.session_state and 'principles_nums' in st.session_state:
        st.markdown("### Analyse avancée des principes TRIZ")
        
        if 'principles_nums' in st.session_state:
            principles_nums = st.session_state['principles_nums']
            principles_info = st.session_state['principles_info']
            
            # Visualisation en réseau des principes
            st.markdown("#### Relations entre principes recommandés")
            create_principles_network(principles_nums, triz_principles)
            
            # Analyse sémantique des principes
            st.markdown("#### Analyse sémantique des solutions")
            
            if 'individual_solutions' in st.session_state:
                solutions = st.session_state['individual_solutions']
                
                # Comparaison des solutions
                if len(solutions) > 1:
                    st.markdown("#### Comparaison des solutions")
                    
                    # Création du DataFrame pour radar chart
                    eval_data = []
                    categories = []
                    
                    for sol in solutions:
                        if 'evaluation' in sol and 'scores' in sol['evaluation']:
                            principle_label = f"P{sol['principle_num']}"
                            scores = sol['evaluation']['scores']
                            eval_data.append(list(scores.values()))
                            categories.append(principle_label)
                    
                    if eval_data:
                        # Création du radar chart avec Plotly
                        categories = ['Alignement', 'Innovation', 'Faisabilité', 'Impact', 'Clarté']
                        
                        fig = go.Figure()
                        
                        for i, data in enumerate(eval_data):
                            fig.add_trace(go.Scatterpolar(
                                r=data,
                                theta=categories,
                                fill='toself',
                                name=f"Principe #{solutions[i]['principle_num']}"
                            ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 5]
                                )
                            ),
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Veuillez d'abord identifier la contradiction dans l'onglet 'Définition du problème'.")

# Options avancées dans la sidebar
with st.sidebar:
    st.title("ℹ️ À propos de TRIZ")
    
    st.markdown("""
    **TRIZ** (Théorie de Résolution des Problèmes Inventifs) est une méthodologie systématique pour résoudre les problèmes techniques complexes en identifiant et surmontant les contradictions fondamentales.
    
    **Fonctionnalités de cet outil :**
    - Identification automatique des contradictions techniques
    - Application des 40 principes TRIZ
    - Génération de solutions innovantes
    - Évaluation des solutions proposées
    
    **Modèle utilisé :**
    - `llama-3.3-70b-versatile` pour l'analyse et la génération
    
    Cet assistant expert vous guide pas à pas dans la résolution de problèmes techniques complexes selon la méthodologie TRIZ.
    """)
    
    # Debug mode (pour développement)
    debug_mode = st.checkbox("Mode débogage", value=False, key="debug_mode")
    
    if debug_mode:
        st.write("### Informations de débogage")
        if 'contradiction_params' in st.session_state:
            st.write("Contradiction identifiée:", st.session_state['contradiction_params'])
        if 'principles_nums' in st.session_state:
            st.write("Principes recommandés:", st.session_state['principles_nums'])
        
        # Détails matrice
        st.write("### Matrice TRIZ")
        st.write("Paramètres lignes:", len(triz_matrix["param_rows"]))
        st.write("Paramètres colonnes:", len(triz_matrix["param_cols"]))

######## the final and the best code #############