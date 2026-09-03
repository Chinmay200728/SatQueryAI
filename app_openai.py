import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import json
import hashlib
import io
from PIL import Image, ImageDraw


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Gemini API key was not found.")
    st.info(
        "Please check that GEMINI_API_KEY is present in your .env file."
    )
    st.stop()

client = genai.Client(
    api_key=GEMINI_API_KEY
)

MODEL = "gemini-2.5-flash"


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="SatQueryAI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "analysis_result": None,
    "analysis_image_id": None,
    "chat_history": [],
    "last_error": None,
    "question_input": "",
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# VISUAL STYLE
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0b0e14;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stMetric"] {
        background-color: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 14px;
    }

    [data-testid="stFileUploader"] {
        border-radius: 16px;
    }

    .small-muted {
        color: #9aa7b7;
        font-size: 0.9rem;
    }

    .status-good {
        color: #55d98a;
        font-weight: 700;
    }

    .status-bad {
        color: #ff7777;
        font-weight: 700;
    }

    .answer-box {
        background: rgba(70, 130, 180, 0.08);
        border: 1px solid rgba(100, 170, 230, 0.18);
        border-radius: 14px;
        padding: 18px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.title("🛰️ SatQueryAI")

st.subheader("AI-Powered Satellite Image Analysis")

st.write(
    "Analyze satellite imagery using AI-powered visual intelligence. "
    "Detect water, vegetation, roads and buildings, estimate coverage "
    "and confidence, understand the surrounding area, and ask questions "
    "about the uploaded image."
)

st.divider()


# ============================================================
# UPLOAD
# ============================================================

st.subheader("📡 Upload a Satellite Image")

st.write(
    "Upload a JPG, PNG, or WEBP satellite image to begin analysis."
)

uploaded_file = st.file_uploader(
    "Choose a satellite image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload a satellite image for AI analysis.",
)


# ============================================================
# RESET WHEN IMAGE CHANGES
# ============================================================

if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    current_image_id = hashlib.md5(
        image_bytes
    ).hexdigest()

    if st.session_state.analysis_image_id != current_image_id:

        st.session_state.analysis_result = None
        st.session_state.analysis_image_id = current_image_id
        st.session_state.chat_history = []
        st.session_state.last_error = None
        st.session_state.question_input = ""


# ============================================================
# JSON SCHEMA
# ============================================================

def make_box_schema():

    return {
        "type": "object",
        "properties": {
            "x1": {
                "type": "number"
            },
            "y1": {
                "type": "number"
            },
            "x2": {
                "type": "number"
            },
            "y2": {
                "type": "number"
            },
        },
        "required": [
            "x1",
            "y1",
            "x2",
            "y2",
        ],
    }


def make_feature_schema():

    return {
        "type": "object",
        "properties": {

            "detected": {
                "type": "boolean"
            },

            "coverage_percent": {
                "type": "number"
            },

            "confidence": {
                "type": "number"
            },

            "explanation": {
                "type": "string"
            },

            "evidence_boxes": {
                "type": "array",
                "items": make_box_schema(),
            },
        },

        "required": [
            "detected",
            "coverage_percent",
            "confidence",
            "explanation",
            "evidence_boxes",
        ],
    }


def make_analysis_schema():

    return {
        "type": "object",

        "properties": {

            "scene_description": {
                "type": "string"
            },

            "area_type": {
                "type": "string",
                "enum": [
                    "Urban",
                    "Suburban",
                    "Rural",
                    "Agricultural",
                    "Natural",
                    "Mixed",
                    "Unclear",
                ],
            },

            "development_level": {
                "type": "string",
                "enum": [
                    "Very Low",
                    "Low",
                    "Moderate",
                    "High",
                    "Very High",
                    "Unclear",
                ],
            },

            "population_density_assessment": {
                "type": "string",
                "enum": [
                    "Very Low",
                    "Low",
                    "Moderate",
                    "High",
                    "Very High",
                    "Unclear",
                ],
            },

            "terrain_description": {
                "type": "string"
            },

            "key_observations": {
                "type": "array",
                "items": {
                    "type": "string"
                },
            },

            "water": make_feature_schema(),

            "vegetation": make_feature_schema(),

            "roads": make_feature_schema(),

            "buildings": make_feature_schema(),
        },

        "required": [

            "scene_description",

            "area_type",

            "development_level",

            "population_density_assessment",

            "terrain_description",

            "key_observations",

            "water",

            "vegetation",

            "roads",

            "buildings",
        ],
    }


# ============================================================
# SAFE NUMBER
# ============================================================

def safe_number(value, default=0):

    try:
        return float(value)

    except Exception:
        return default


# ============================================================
# SATELLITE ANALYSIS
# ============================================================

def analyze_satellite_image(
    image_bytes,
    mime_type,
):

    prompt = """
You are a professional satellite and remote-sensing image analyst.

Analyze ONLY visible evidence in the supplied satellite image.

Detect these four features:

1. Water
2. Vegetation
3. Roads
4. Buildings

For each feature return:

- detected: true or false
- coverage_percent: approximate percentage of the TOTAL image area
- confidence: 0 to 100
- explanation: concise evidence-based explanation
- evidence_boxes: boxes around clearly visible examples

Bounding boxes use normalized coordinates from 0 to 1000.

Origin is the TOP-LEFT.

x1 = left
y1 = top
x2 = right
y2 = bottom

Only create boxes for clearly visible evidence.

Do not invent objects.

WATER:

Only classify visible lakes, rivers, reservoirs, ponds,
canals, sea or coastal water.

Do not classify shadows, dark vegetation,
asphalt or rooftops as water.

VEGETATION:

Include clearly visible trees, forest, grass,
crops, plantations and other vegetation.

ROADS:

Only classify recognizable roads and road corridors.

Do not classify rivers, field boundaries,
shadows or building edges as roads.

BUILDINGS:

Only classify clearly visible structures.

If a feature is not reliably visible:

detected = false
coverage_percent = 0
evidence_boxes = []

Coverage percentages are estimates and do not need
to total 100%.

Also determine:

scene_description

area_type:
Urban, Suburban, Rural, Agricultural,
Natural, Mixed or Unclear

development_level:
Very Low, Low, Moderate, High,
Very High or Unclear

population_density_assessment:
Very Low, Low, Moderate, High,
Very High or Unclear

This is only a visual assessment.
Do NOT claim actual population numbers.

terrain_description

key_observations:
3 to 5 concise observations based only
on visible evidence.
"""

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type,
    )

    response = client.models.generate_content(
        model=MODEL,

        contents=[
            image_part,
            prompt,
        ],

        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=make_analysis_schema(),
            temperature=0.2,
        ),
    )

    if not response.text:

        raise RuntimeError(
            "Gemini returned an empty analysis."
        )

    return json.loads(response.text)


# ============================================================
# FEATURE CARD
# ============================================================

def show_feature(
    icon,
    name,
    data,
):

    detected = bool(
        data.get(
            "detected",
            False,
        )
    )

    coverage = max(
        0,
        min(
            100,
            safe_number(
                data.get(
                    "coverage_percent",
                    0,
                )
            ),
        ),
    )

    confidence = max(
        0,
        min(
            100,
            safe_number(
                data.get(
                    "confidence",
                    0,
                )
            ),
        ),
    )

    explanation = data.get(
        "explanation",
        "No explanation available.",
    )

    with st.container(border=True):

        st.markdown(
            f"### {icon} {name}"
        )

        if detected:

            st.success(
                "✅ DETECTED"
            )

        else:

            st.error(
                "❌ NOT DETECTED"
            )

        metric1, metric2 = st.columns(2)

        with metric1:

            st.metric(
                "Coverage",
                f"{coverage:.1f}%",
            )

        with metric2:

            st.metric(
                "Confidence",
                f"{confidence:.0f}%",
            )

        st.write(
            explanation
        )


# ============================================================
# MARKED / EVIDENCE IMAGE
# ============================================================

def create_evidence_image(
    image_bytes,
    analysis_result,
):

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGBA")

    width, height = image.size

    # --------------------------------------------------------
    # Transparent overlay layer
    # --------------------------------------------------------

    overlay = Image.new(
        "RGBA",
        image.size,
        (0, 0, 0, 0),
    )

    draw = ImageDraw.Draw(
        overlay,
        "RGBA",
    )

    settings = [

        (
            "water",
            "WATER",
            (0, 150, 255),
        ),

        (
            "vegetation",
            "VEGETATION",
            (0, 210, 80),
        ),

        (
            "roads",
            "ROADS",
            (255, 165, 0),
        ),

        (
            "buildings",
            "BUILDINGS",
            (255, 55, 65),
        ),
    ]

    found_features = []

    marked_counts = {
        "water": 0,
        "vegetation": 0,
        "roads": 0,
        "buildings": 0,
    }

    for key, label, color in settings:

        data = analysis_result.get(
            key,
            {},
        )

        boxes = data.get(
            "evidence_boxes",
            [],
        )

        if not isinstance(
            boxes,
            list,
        ):
            continue

        if boxes:

            found_features.append(
                label
            )

        for box in boxes:

            try:

                x1 = max(
                    0,
                    min(
                        1000,
                        float(
                            box.get(
                                "x1",
                                0,
                            )
                        ),
                    ),
                )

                y1 = max(
                    0,
                    min(
                        1000,
                        float(
                            box.get(
                                "y1",
                                0,
                            )
                        ),
                    ),
                )

                x2 = max(
                    0,
                    min(
                        1000,
                        float(
                            box.get(
                                "x2",
                                0,
                            )
                        ),
                    ),
                )

                y2 = max(
                    0,
                    min(
                        1000,
                        float(
                            box.get(
                                "y2",
                                0,
                            )
                        ),
                    ),
                )

                left = int(
                    min(x1, x2)
                    / 1000
                    * width
                )

                top = int(
                    min(y1, y2)
                    / 1000
                    * height
                )

                right = int(
                    max(x1, x2)
                    / 1000
                    * width
                )

                bottom = int(
                    max(y1, y2)
                    / 1000
                    * height
                )

                if right - left < 10:
                    continue

                if bottom - top < 10:
                    continue

                marked_counts[key] += 1

                # ------------------------------------------------
                # TRANSPARENT FILL
                # ------------------------------------------------

                draw.rectangle(
                    [
                        left,
                        top,
                        right,
                        bottom,
                    ],

                    fill=(
                        color[0],
                        color[1],
                        color[2],
                        55,
                    ),

                    outline=(
                        color[0],
                        color[1],
                        color[2],
                        235,
                    ),

                    width=max(
                        3,
                        int(
                            min(
                                width,
                                height,
                            ) / 450
                        ),
                    ),
                )

                # ------------------------------------------------
                # LABEL
                # ------------------------------------------------

                label_height = 24

                label_width = max(
                    90,
                    len(label) * 8 + 18,
                )

                label_left = left

                label_top = max(
                    0,
                    top - label_height,
                )

                label_right = min(
                    width,
                    left + label_width,
                )

                label_bottom = top

                # Slightly transparent label
                draw.rounded_rectangle(
                    [
                        label_left,
                        label_top,
                        label_right,
                        label_bottom,
                    ],

                    radius=5,

                    fill=(
                        color[0],
                        color[1],
                        color[2],
                        190,
                    ),
                )

                try:

                    draw.text(
                        (
                            left + 7,
                            label_top + 4,
                        ),
                        label,
                        fill=(
                            255,
                            255,
                            255,
                            255,
                        ),
                    )

                except Exception:
                    pass

            except Exception:
                continue

    # --------------------------------------------------------
    # Combine original image + transparent overlay
    # --------------------------------------------------------

    final_image = Image.alpha_composite(
        image,
        overlay,
    )

    final_image = final_image.convert(
        "RGB"
    )

    return (
        final_image,
        found_features,
        marked_counts,
    )


# ============================================================
# HORIZONTAL GRAPH
# ============================================================

def horizontal_chart(
    title,
    values,
    description,
):

    st.subheader(title)

    st.caption(
        description
    )

    # IMPORTANT:
    # Use Streamlit native components.
    # Do NOT print HTML code.
    # This prevents the code-looking screen
    # that appeared in your screenshot.

    for name, value in values:

        value = max(
            0,
            min(
                100,
                safe_number(value),
            ),
        )

        st.write(
            f"**{name}** — {value:.1f}%"
        )

        st.progress(
            int(round(value)),
        )

    st.write("")


# ============================================================
# ASK AI
# ============================================================

def ask_about_image(
    image_bytes,
    mime_type,
    question,
):

    prompt = f"""
You are an expert satellite-image analysis assistant.

Answer the user's question using ONLY visible evidence
from the supplied satellite image.

Do not invent information.

If something cannot be determined from the image,
say that clearly.

Give a concise but useful answer.

User question:

{question}
"""

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type,
    )

    response = client.models.generate_content(
        model=MODEL,

        contents=[
            image_part,
            prompt,
        ],

        config=types.GenerateContentConfig(
            temperature=0.2,
        ),
    )

    if not response.text:

        raise RuntimeError(
            "Gemini returned an empty answer."
        )

    return response.text


# ============================================================
# NO IMAGE
# ============================================================

if uploaded_file is None:

    st.info(
        "👆 Upload a satellite image above to begin."
    )

    st.divider()

    st.subheader(
        "What SatQueryAI can analyze"
    )

    a, b, c, d = st.columns(4)

    with a:

        st.metric(
            "💧 Water",
            "Detection",
        )

    with b:

        st.metric(
            "🌳 Vegetation",
            "Detection",
        )

    with c:

        st.metric(
            "🛣️ Roads",
            "Detection",
        )

    with d:

        st.metric(
            "🏢 Buildings",
            "Detection",
        )


# ============================================================
# IMAGE AVAILABLE
# ============================================================

else:

    st.image(
        uploaded_file,
        caption="Uploaded Satellite Image",
        use_container_width=True,
    )

    st.divider()

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    if st.button(
        "🔎 Analyze Image with AI",
        type="primary",
        use_container_width=True,
    ):

        st.session_state.last_error = None

        with st.spinner(
            "🛰️ AI is analyzing the satellite image..."
        ):

            try:

                image_bytes = (
                    uploaded_file.getvalue()
                )

                mime_type = (
                    uploaded_file.type
                    or "image/jpeg"
                )

                result = analyze_satellite_image(
                    image_bytes,
                    mime_type,
                )

                st.session_state.analysis_result = (
                    result
                )

                st.success(
                    "✅ Analysis completed successfully."
                )

            except Exception as e:

                st.session_state.last_error = str(e)

                st.error(
                    "❌ The image analysis could not be completed."
                )

                st.info(
                    "Please check your Gemini API key, "
                    "internet connection, image format, "
                    "and Gemini model availability."
                )

                with st.expander(
                    "Technical error details"
                ):

                    st.code(
                        str(e)
                    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    result = (
        st.session_state.analysis_result
    )

    if result is not None:

        # ====================================================
        # OVERVIEW
        # ====================================================

        st.divider()

        st.header(
            "🌍 AI Interpretation"
        )

        st.write(
            result.get(
                "scene_description",
                "No scene description available.",
            )
        )

        # ====================================================
        # AREA INTELLIGENCE
        # ====================================================

        st.header(
            "📍 Area Intelligence"
        )

        area1, area2, area3 = st.columns(3)

        with area1:

            st.metric(
                "Area Type",
                result.get(
                    "area_type",
                    "Unclear",
                ),
            )

        with area2:

            st.metric(
                "Development",
                result.get(
                    "development_level",
                    "Unclear",
                ),
            )

        with area3:

            st.metric(
                "Visual Population Density",
                result.get(
                    "population_density_assessment",
                    "Unclear",
                ),
            )

        st.write(
            "**Terrain:** "
            + result.get(
                "terrain_description",
                "Unclear",
            )
        )

        observations = result.get(
            "key_observations",
            [],
        )

        if observations:

            st.subheader(
                "🔍 Key Observations"
            )

            for observation in observations:

                st.write(
                    "• " + observation
                )

        # ====================================================
        # FEATURE DETECTION
        # ====================================================

        st.divider()

        st.header(
            "🔎 Feature Detection"
        )

        col1, col2 = st.columns(2)

        with col1:

            show_feature(
                "💧",
                "Water",
                result.get(
                    "water",
                    {},
                ),
            )

            show_feature(
                "🛣️",
                "Roads",
                result.get(
                    "roads",
                    {},
                ),
            )

        with col2:

            show_feature(
                "🌳",
                "Vegetation",
                result.get(
                    "vegetation",
                    {},
                ),
            )

            show_feature(
                "🏢",
                "Buildings",
                result.get(
                    "buildings",
                    {},
                ),
            )

        # ====================================================
        # AI ANALYZED & MARKED MAP
        # ====================================================

        st.divider()

        st.header(
            "🗺️ AI Analyzed & Marked Map"
        )

        st.write(
            "The colored transparent boxes show areas "
            "where the AI identified roads, buildings, "
            "vegetation or water."
        )

        try:

            (
                evidence_image,
                legend,
                marked_counts,
            ) = create_evidence_image(
                uploaded_file.getvalue(),
                result,
            )

            st.image(
                evidence_image,
                caption="AI Analyzed & Marked Satellite Image",
                use_container_width=True,
            )

            # ------------------------------------------------
            # LEGEND
            # ------------------------------------------------

            st.markdown(
                """
                **Color Legend**

                🔵 **Water** &nbsp;&nbsp;
                🟢 **Vegetation** &nbsp;&nbsp;
                🟠 **Roads** &nbsp;&nbsp;
                🔴 **Buildings**
                """
            )

            # ------------------------------------------------
            # DOWNLOAD BUTTON
            # ------------------------------------------------

            image_buffer = io.BytesIO()

            evidence_image.save(
                image_buffer,
                format="PNG",
            )

            st.download_button(
                "⬇️ Save Marked Image",
                data=image_buffer.getvalue(),
                file_name="satqueryai_marked_map.png",
                mime="image/png",
                use_container_width=True,
            )

        except Exception as e:

            st.warning(
                "The marked map could not be generated."
            )

            with st.expander(
                "Marked map technical details"
            ):

                st.code(
                    str(e)
                )

        # ====================================================
        # MARKED AREAS
        # ====================================================

        st.header(
            "📍 Marked Areas"
        )

        water_count = marked_counts.get(
            "water",
            0,
        )

        vegetation_count = marked_counts.get(
            "vegetation",
            0,
        )

        roads_count = marked_counts.get(
            "roads",
            0,
        )

        buildings_count = marked_counts.get(
            "buildings",
            0,
        )

        if water_count > 0:

            st.write(
                f"💧 **Water:** "
                f"{water_count} marked region(s)"
            )

        else:

            st.write(
                "💧 **Water:** Not detected."
            )

        if vegetation_count > 0:

            st.write(
                f"🌳 **Vegetation:** "
                f"{vegetation_count} marked region(s)"
            )

        else:

            st.write(
                "🌳 **Vegetation:** Not detected."
            )

        if roads_count > 0:

            st.write(
                f"🛣️ **Roads:** "
                f"{roads_count} marked region(s)"
            )

        else:

            st.write(
                "🛣️ **Roads:** Not detected."
            )

        if buildings_count > 0:

            st.write(
                f"🏢 **Buildings:** "
                f"{buildings_count} marked region(s)"
            )

        else:

            st.write(
                "🏢 **Buildings:** Not detected."
            )

        # ====================================================
        # ANALYTICS
        # ====================================================

        st.divider()

        st.header(
            "📊 Visual Analytics"
        )

        # ----------------------------------------------------
        # FEATURE COVERAGE
        # ----------------------------------------------------

        coverage_values = [

            (
                "Water",
                result.get(
                    "water",
                    {},
                ).get(
                    "coverage_percent",
                    0,
                ),
            ),

            (
                "Vegetation",
                result.get(
                    "vegetation",
                    {},
                ).get(
                    "coverage_percent",
                    0,
                ),
            ),

            (
                "Roads",
                result.get(
                    "roads",
                    {},
                ).get(
                    "coverage_percent",
                    0,
                ),
            ),

            (
                "Buildings",
                result.get(
                    "buildings",
                    {},
                ).get(
                    "coverage_percent",
                    0,
                ),
            ),
        ]

        horizontal_chart(
            "📊 Feature Coverage",
            coverage_values,
            "Estimated percentage of the total image area occupied by each feature.",
        )

        # ----------------------------------------------------
        # AI CONFIDENCE
        # ----------------------------------------------------

        confidence_values = [

            (
                "Water",
                result.get(
                    "water",
                    {},
                ).get(
                    "confidence",
                    0,
                ),
            ),

            (
                "Vegetation",
                result.get(
                    "vegetation",
                    {},
                ).get(
                    "confidence",
                    0,
                ),
            ),

            (
                "Roads",
                result.get(
                    "roads",
                    {},
                ).get(
                    "confidence",
                    0,
                ),
            ),

            (
                "Buildings",
                result.get(
                    "buildings",
                    {},
                ).get(
                    "confidence",
                    0,
                ),
            ),
        ]

        horizontal_chart(
            "🤖 AI Confidence",
            confidence_values,
            "How strongly the visible evidence supports each detection.",
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        st.divider()

        st.header(
            "📋 Analysis Summary"
        )

        summary = []

        features = [

            (
                "💧",
                "Water",
                result.get(
                    "water",
                    {},
                ),
            ),

            (
                "🌳",
                "Vegetation",
                result.get(
                    "vegetation",
                    {},
                ),
            ),

            (
                "🛣️",
                "Roads",
                result.get(
                    "roads",
                    {},
                ),
            ),

            (
                "🏢",
                "Buildings",
                result.get(
                    "buildings",
                    {},
                ),
            ),
        ]

        for icon, name, data in features:

            detected = bool(
                data.get(
                    "detected",
                    False,
                )
            )

            coverage = safe_number(
                data.get(
                    "coverage_percent",
                    0,
                )
            )

            confidence = safe_number(
                data.get(
                    "confidence",
                    0,
                )
            )

            summary.append(
                {
                    "Feature": (
                        f"{icon} {name}"
                    ),

                    "Status": (
                        "Detected"
                        if detected
                        else "Not Detected"
                    ),

                    "Coverage": (
                        f"{coverage:.1f}%"
                    ),

                    "Confidence": (
                        f"{confidence:.0f}%"
                    ),
                }
            )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
        )

        # ====================================================
        # ASK AI
        # ====================================================

        st.divider()

        st.header(
            "💬 Ask About This Satellite Image"
        )

        st.write(
            "Ask questions about the same uploaded satellite image."
        )

        q1, q2, q3 = st.columns(3)

        if q1.button(
            "🏙️ Is this urban?",
            use_container_width=True,
        ):

            st.session_state.question_input = (
                "Does this appear to be an urban area? "
                "Explain the visible evidence."
            )

        if q2.button(
            "🛣️ Are there many roads?",
            use_container_width=True,
        ):

            st.session_state.question_input = (
                "Are there many visible roads in this image?"
            )

        if q3.button(
            "🌳 Describe vegetation",
            use_container_width=True,
        ):

            st.session_state.question_input = (
                "Describe the visible vegetation and "
                "where it appears."
            )

        question = st.text_input(
            "Your question",
            value=st.session_state.question_input,
            placeholder=(
                "Ask something about this satellite image..."
            ),
        )

        if st.button(
            "💬 Ask AI",
            type="primary",
            use_container_width=True,
        ):

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                with st.spinner(
                    "🤖 AI is examining the image..."
                ):

                    try:

                        answer = ask_about_image(
                            uploaded_file.getvalue(),
                            uploaded_file.type
                            or "image/jpeg",
                            question.strip(),
                        )

                        st.session_state.chat_history.append(
                            {
                                "question": (
                                    question.strip()
                                ),
                                "answer": answer,
                            }
                        )

                        st.session_state.question_input = ""

                    except Exception as e:

                        st.error(
                            "❌ The AI could not answer "
                            "that question right now."
                        )

                        with st.expander(
                            "Technical error details"
                        ):

                            st.code(
                                str(e)
                            )

        # ====================================================
        # CHAT HISTORY
        # ====================================================

        if st.session_state.chat_history:

            st.subheader(
                "🗨️ Conversation"
            )

            for item in (
                st.session_state.chat_history
            ):

                with st.chat_message(
                    "user"
                ):

                    st.write(
                        item["question"]
                    )

                with st.chat_message(
                    "assistant"
                ):

                    st.write(
                        item["answer"]
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🛰️ SatQueryAI • AI-powered satellite image analysis • "
    "Visual evidence • Area intelligence • Interactive AI"
)