import streamlit as st
import pandas as pd
import numpy as np
import altair as alt


# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="Lifestyle & Comorbidity Dashboard",
    layout="wide"
)

DATA_PATH = "processed_data.csv"

OUTCOME_COLS = [
    "Heart_Attack",
    "Diabetes",
    "Stroke",
    "Celiac_Disease",
    "Arthritis",
    "Liver_Disease",
    "Asthma",
    "COPD",
]

LIFESTYLE_COLS = [
    "Cigarettes_Per_Day",
    "Sleep_Hours",
    "Alcohol_Use_Frequency",
    "Physical_Activity_Equivalent_Min",
]

METABOLIC_COLS = [
    "BMI",
    "Waist_Circumference",
    "Total_Cholesterol",
    "HDL_Cholesterol",
    "LDL_Cholesterol",
    "Triglycerides",
    "Serum_Glucose",
    "Glycohemoglobin",
    "Platelet_Count",
    "Hemoglobin",
    "Hematocrit",
    "Iron_Saturation",
    "Serum_Iron",
    "Creatinine",
    "Uric_Acid",
    "WBC_Count",
    "Diastolic_BP_Average",
    "Systolic_BP_Average",
]

DEMOGRAPHIC_COLS = [
    "Age",
    "Gender",
    "Ethnicity",
    "Education_Level",
    "Income_to_Poverty_Ratio",
]


# -----------------------------
# Helpers
# -----------------------------
def nice_label(name: str) -> str:
    return name.replace("_", " ")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Ensure outcomes are strings
    for col in OUTCOME_COLS:
        df[col] = df[col].astype(str)

    # Create comorbidity summary variables
    outcome_yes = df[OUTCOME_COLS].apply(lambda c: c.str.upper().eq("YES"))
    df["Comorbidity_Count"] = outcome_yes.sum(axis=1)
    df["Any_Comorbidity"] = np.where(df["Comorbidity_Count"] > 0, "Yes", "No")

    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    # -------- Demographics ----------
    st.sidebar.subheader("Demographics")

    # Age
    age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
    age_range = st.sidebar.slider(
        "Age range",
        min_value=age_min,
        max_value=age_max,
        value=(age_min, age_max),
        step=1,
    )

    # Ethnicity
    ethnicities = sorted(df["Ethnicity"].dropna().unique())
    selected_ethnicities = st.sidebar.multiselect(
        "Ethnicity",
        options=ethnicities,
        default=ethnicities,
    )

    # Education
    edu_levels = sorted(df["Education_Level"].dropna().unique())
    selected_edu = st.sidebar.multiselect(
        "Education level",
        options=edu_levels,
        default=edu_levels,
    )

    # Income-to-poverty ratio
    inc_min, inc_max = float(df["Income_to_Poverty_Ratio"].min()), float(df["Income_to_Poverty_Ratio"].max())
    income_range = st.sidebar.slider(
        "Income-to-poverty ratio",
        min_value=float(round(inc_min, 1)),
        max_value=float(round(inc_max, 1)),
        value=(float(round(inc_min, 1)), float(round(inc_max, 1))),
        step=0.1,
    )

    # -------- Lifestyle ----------
    st.sidebar.subheader("Lifestyle")

    # Cigarettes per day
    c_min, c_max = int(df["Cigarettes_Per_Day"].min()), int(df["Cigarettes_Per_Day"].max())
    cig_range = st.sidebar.slider(
        "Cigarettes per day",
        min_value=c_min,
        max_value=c_max,
        value=(c_min, c_max),
        step=1,
    )

    # Sleep hours
    s_min, s_max = int(df["Sleep_Hours"].min()), int(df["Sleep_Hours"].max())
    sleep_range = st.sidebar.slider(
        "Sleep hours",
        min_value=s_min,
        max_value=s_max,
        value=(s_min, s_max),
        step=1,
    )

    # Alcohol use frequency
    a_min, a_max = int(df["Alcohol_Use_Frequency"].min()), int(df["Alcohol_Use_Frequency"].max())
    alcohol_range = st.sidebar.slider(
        "Alcohol use frequency (code)",
        min_value=a_min,
        max_value=a_max,
        value=(a_min, a_max),
        step=1,
    )

    # Physical activity
    p_min, p_max = int(df["Physical_Activity_Equivalent_Min"].min()), int(df["Physical_Activity_Equivalent_Min"].max())
    pa_range = st.sidebar.slider(
        "Physical activity (equivalent minutes per week)",
        min_value=p_min,
        max_value=p_max,
        value=(p_min, p_max),
        step=10,
    )

    # Build mask (no Gender filter here)
    mask = (
        df["Age"].between(age_range[0], age_range[1])
        & df["Income_to_Poverty_Ratio"].between(income_range[0], income_range[1])
        & df["Cigarettes_Per_Day"].between(cig_range[0], cig_range[1])
        & df["Sleep_Hours"].between(sleep_range[0], sleep_range[1])
        & df["Alcohol_Use_Frequency"].between(alcohol_range[0], alcohol_range[1])
        & df["Physical_Activity_Equivalent_Min"].between(pa_range[0], pa_range[1])
    )

    if selected_ethnicities:
        mask &= df["Ethnicity"].isin(selected_ethnicities)
    if selected_edu:
        mask &= df["Education_Level"].isin(selected_edu)

    return df[mask].copy()


def prevalence_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in OUTCOME_COLS:
        s = df[col].str.upper()
        prev = (s == "YES").mean() * 100
        rows.append(
            {"Outcome": nice_label(col), "Prevalence (%)": round(prev, 1)}
        )
    prev_df = pd.DataFrame(rows)
    return prev_df.sort_values("Prevalence (%)", ascending=False)




# -----------------------------
# New from Shu:
# -----------------------------


# -----------------------------
# Main app
# -----------------------------
df = load_data(DATA_PATH)

st.title("Cardiometabolic Risk Explorer Dashboard")
st.markdown(
    """
Interactive dashboard to explore how age, sex, lifestyle behaviours and biomarkers 
relate to cardiometabolic risk and multimorbidity.
    
1. Age-stratified risk-factor section:
    Visualizes how risk factors change with age and relate to cardiometabolic health.

2. Correlation matrix of biomarkers:
    Examines pairwise correlations among metabolic and cardiovascular biomarkers, with filters for age, sex, and disease state.

3. Disease prevalence summary:
    Summarizes comorbidity prevalence under current filters.

4. Metabolic markers across comorbidity levels:
    Compares metabolic markers across varying comorbidity burdens.

5. Comorbidity / disease prevalence by lifestyle:
    Shows how comorbidity prevalence shifts across lifestyle patterns.   
"""
)

filtered = apply_filters(df)
st.markdown(f"### Current selection: {len(filtered):,} participants")

if filtered.empty:
    st.warning("No participants match the current filter settings. Try relaxing your filters.")
    st.stop()



# -----------------------------
# 1. Age-stratified risk-factor section (Shu)
# -----------------------------

def add_age_band_column(df: pd.DataFrame, band_width: int = 10) -> pd.DataFrame:
    """Add a categorical Age_Band column with width-year bands."""
    tmp = df.copy()
    age_min = int(np.floor(tmp["Age"].min() / band_width) * band_width)
    age_max = int(np.ceil(tmp["Age"].max() / band_width) * band_width)

    # Build bins and labels like "40–49"
    bins = list(range(age_min, age_max + band_width, band_width))
    labels = [f"{b}–{b + band_width - 1}" for b in bins[:-1]]

    tmp["Age_Band"] = pd.cut(
        tmp["Age"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    return tmp


def age_stratified_section(filtered: pd.DataFrame) -> None:
    st.subheader("1. Age-stratified distribution of risk factors")

    # Risk factor selector (metabolic + lifestyle)
    risk_options = METABOLIC_COLS + LIFESTYLE_COLS
    risk_var = st.selectbox(
        "Risk factor",
        options=risk_options,
        index=risk_options.index("BMI") if "BMI" in risk_options else 0,
        format_func=nice_label,
    )

    # Age display toggle
    age_mode = st.radio(
        "Age representation",
        options=["Grouped (age bands)", "Continuous age"],
        index=0,
        horizontal=True,
    )

    # Add age bands & highlight column
    tmp = add_age_band_column(filtered)
    age_bands = [b for b in tmp["Age_Band"].dropna().unique().tolist()]

    highlight_choice = st.selectbox(
        "Highlight age band (optional)",
        options=["All ages"] + age_bands,
        index=0,
        help="Selected band is emphasized in all plots; others are faded.",
    )

    if highlight_choice == "All ages":
        tmp["Is_Highlighted"] = "Highlighted"
    else:
        # Age_Band is a categorical; compare as string
        tmp["Is_Highlighted"] = np.where(
            tmp["Age_Band"].astype(str) == str(highlight_choice),
            "Highlighted",
            "Other",
        )

    # ----------------- top row: distribution by age -----------------
    top_left, top_right = st.columns(2)

    # Top-left: distribution of risk factor across age
    if age_mode.startswith("Grouped"):
        # Boxplots by age band
        box_chart = (
            alt.Chart(tmp.dropna(subset=["Age_Band"]))
            .mark_boxplot()
            .encode(
                x=alt.X("Age_Band:N", title="Age band"),
                y=alt.Y(f"{risk_var}:Q", title=nice_label(risk_var)),
                opacity=alt.condition(
                    "datum.Is_Highlighted == 'Highlighted'",
                    alt.value(1.0),
                    alt.value(0.2),
                ),
                tooltip=[
                    alt.Tooltip("Age_Band:N", title="Age band"),
                    alt.Tooltip(f"{risk_var}:Q", title=nice_label(risk_var), aggregate="mean"),
                    alt.Tooltip("count()", title="n"),
                ],
            )
            .properties(height=300)
        )
    else:
        # Continuous age: scatter + smooth line
        base = alt.Chart(tmp).encode(
            x=alt.X("Age:Q", title="Age"),
            y=alt.Y(f"{risk_var}:Q", title=nice_label(risk_var)),
        )

        points = base.mark_circle(size=20, opacity=0.3).encode(
            opacity=alt.condition(
                "datum.Is_Highlighted == 'Highlighted'",
                alt.value(0.7),
                alt.value(0.15),
            ),
            color=alt.Color("Age_Band:N", title="Age band"),
            tooltip=["Age:Q", f"{risk_var}:Q", "Age_Band:N"],
        )

        smooth = base.transform_loess(
            "Age", risk_var, groupby=["Age_Band"], bandwidth=0.5
        ).mark_line(size=2).encode(
            color=alt.Color("Age_Band:N", title="Age band"),
            opacity=alt.condition(
                "datum.Is_Highlighted == 'Highlighted'",
                alt.value(1.0),
                alt.value(0.2),
            ),
        )

        box_chart = (points + smooth).properties(height=300)

    with top_left:
        st.markdown("**Distribution by age**")
        st.altair_chart(box_chart, use_container_width=True)

    # Top-right: density “ridgeline-style” across age bands
    density_chart = (
        alt.Chart(tmp.dropna(subset=["Age_Band"]))
        .transform_density(
            risk_var,
            groupby=["Age_Band"],
            as_=[risk_var, "density"],
        )
        .mark_line()
        .encode(
            x=alt.X(f"{risk_var}:Q", title=nice_label(risk_var)),
            y=alt.Y("density:Q", title="Density"),
            color=alt.Color("Age_Band:N", title="Age band"),
            opacity=alt.condition(
                "datum.Is_Highlighted == 'Highlighted'",
                alt.value(1.0),
                alt.value(0.25),
            ),
            tooltip=[
                alt.Tooltip("Age_Band:N", title="Age band"),
                alt.Tooltip(f"{risk_var}:Q", title=nice_label(risk_var)),
                alt.Tooltip("density:Q", title="Density", format=".3f"),
            ],
        )
        .properties(height=300)
    )

    with top_right:
        st.markdown("**Risk-factor distribution by age band**")
        st.altair_chart(density_chart, use_container_width=True)

    # ----------------- bottom row: by gender & CV correlation -----------------
    bottom_left, bottom_right = st.columns(2)

    # Bottom-left: mean risk factor vs age, by gender
    # Aggregate to keep it smooth-ish
    gender_age = (
        tmp.dropna(subset=["Gender"])
        .groupby(["Age", "Gender"], as_index=False)[risk_var]
        .mean()
    )
    # Add bands & highlight info to aggregated df
    gender_age = add_age_band_column(gender_age)
    if highlight_choice == "All ages":
        gender_age["Is_Highlighted"] = "Highlighted"
    else:
        gender_age["Is_Highlighted"] = np.where(
            gender_age["Age_Band"].astype(str) == str(highlight_choice),
            "Highlighted",
            "Other",
        )

    gender_chart = (
        alt.Chart(gender_age)
        .mark_line()
        .encode(
            x=alt.X("Age:Q", title="Age"),
            y=alt.Y(f"{risk_var}:Q", title=f"Mean {nice_label(risk_var)}"),
            color=alt.Color("Gender:N", title="Gender"),
            opacity=alt.condition(
                "datum.Is_Highlighted == 'Highlighted'",
                alt.value(1.0),
                alt.value(0.3),
            ),
            tooltip=["Age:Q", "Gender:N", f"{risk_var}:Q"],
        )
        .properties(height=280)
    )

    with bottom_left:
        st.markdown("**By gender across age**")
        st.altair_chart(gender_chart, use_container_width=True)

    # Bottom-right: correlation with a cardiometabolic variable
    cardio_candidates = [
        "Systolic_BP_Average",
        "Diastolic_BP_Average",
        "Total_Cholesterol",
        "LDL_Cholesterol",
        "HDL_Cholesterol",
        "Triglycerides",
        "Serum_Glucose",
        "Glycohemoglobin",
    ]
    cardio_candidates = [c for c in cardio_candidates if c in METABOLIC_COLS]

    cv_var = st.selectbox(
        "Cardiometabolic variable (y-axis)",
        options=cardio_candidates,
        index=cardio_candidates.index("Systolic_BP_Average")
        if "Systolic_BP_Average" in cardio_candidates
        else 0,
        format_func=nice_label,
    )

    corr_chart = (
        alt.Chart(tmp.dropna(subset=["Age_Band"]))
        .mark_circle(size=25)
        .encode(
            x=alt.X(f"{risk_var}:Q", title=nice_label(risk_var)),
            y=alt.Y(f"{cv_var}:Q", title=nice_label(cv_var)),
            color=alt.Color("Age_Band:N", title="Age band"),
            opacity=alt.condition(
                "datum.Is_Highlighted == 'Highlighted'",
                alt.value(0.9),
                alt.value(0.2),
            ),
            tooltip=[
                alt.Tooltip("Age:Q", title="Age"),
                alt.Tooltip("Age_Band:N", title="Age band"),
                alt.Tooltip(f"{risk_var}:Q", title=nice_label(risk_var)),
                alt.Tooltip(f"{cv_var}:Q", title=nice_label(cv_var)),
            ],
        )
        .properties(height=280)
    )

    with bottom_right:
        st.markdown("**Correlation with cardiometabolic health**")
        st.altair_chart(corr_chart, use_container_width=True)

# -----------------------------
# 2. Correlation matrix of biomarkers (Shu)
# -----------------------------
def correlation_matrix_section(filtered: pd.DataFrame) -> None:
    st.subheader("2. Correlation matrix of biomarkers")

    df = add_age_band_column(filtered)

    # --- Controls ---
    # Age bands
    age_bands = sorted(df["Age_Band"].dropna().unique().tolist())
    selected_bands = st.multiselect(
        "Age bands",
        options=age_bands,
        default=age_bands,
    )
    if selected_bands:
        df = df[df["Age_Band"].isin(selected_bands)]

    # Gender
    gender_vals = sorted(df["Gender"].dropna().astype(str).unique().tolist())
    selected_gender = st.multiselect(
        "Gender",
        options=gender_vals,
        default=gender_vals,
    )
    if selected_gender:
        df = df[df["Gender"].astype(str).isin(selected_gender)]

    # Disease state
    disease_options = (
        ["All participants", "Healthy (no comorbidities)", "Any comorbidity (≥1)"]
        + [nice_label(c) for c in OUTCOME_COLS]
    )
    disease_choice = st.selectbox("Disease state", options=disease_options)

    if disease_choice == "Healthy (no comorbidities)":
        df = df[df["Any_Comorbidity"] == "No"]
    elif disease_choice == "Any comorbidity (≥1)":
        df = df[df["Any_Comorbidity"] == "Yes"]
    elif disease_choice != "All participants":
        # Map pretty label back to column
        col_map = {nice_label(c): c for c in OUTCOME_COLS}
        disease_col = col_map[disease_choice]
        df = df[df[disease_col].astype(str).str.upper() == "YES"]

    # If nothing left, bail
    if df.empty:
        st.warning("No data available under the current age / gender / disease filters.")
        return

    # Biomarkers to include
    default_biomarkers = [
        "BMI",
        "Total_Cholesterol",
        "HDL_Cholesterol",
        "LDL_Cholesterol",
        "Triglycerides",
        "Serum_Glucose",
        "Glycohemoglobin",
        "Systolic_BP_Average",
        "Diastolic_BP_Average",
    ]
    default_biomarkers = [b for b in default_biomarkers if b in METABOLIC_COLS]

    biomarkers = st.multiselect(
        "Biomarkers to include in the correlation matrix",
        options=METABOLIC_COLS,
        default=default_biomarkers or METABOLIC_COLS[:6],
        format_func=nice_label,
    )

    if len(biomarkers) < 2:
        st.info("Select at least two biomarkers to compute correlations.")
        return

    df_bio = df[biomarkers].dropna()
    if df_bio.empty:
        st.warning("No complete biomarker measurements for the current filters.")
        return

    # --- Build pairwise long data and correlations ---
    from itertools import combinations

    long_records = []
    for x, y in combinations(biomarkers, 2):
        sub = df_bio[[x, y]].dropna()
        if sub.empty:
            continue
        for xv, yv in sub.itertuples(index=False):
            long_records.append(
                {
                    "Biomarker_X": x,
                    "Biomarker_Y": y,
                    "x_value": xv,
                    "y_value": yv,
                }
            )

    if not long_records:
        st.warning("Not enough overlapping data across biomarker pairs.")
        return

    long_df = pd.DataFrame(long_records)

    # Compute correlations per pair
    corr_rows = []
    for (x, y), g in long_df.groupby(["Biomarker_X", "Biomarker_Y"]):
        r = g[["x_value", "y_value"]].corr().iloc[0, 1]
        corr_rows.append(
            {"Biomarker_X": x, "Biomarker_Y": y, "Correlation": r}
        )

    corr_df = pd.DataFrame(corr_rows)

    # Mirror to get a full matrix (optional but looks nicer)
    mirror_df = corr_df.rename(
        columns={"Biomarker_X": "Biomarker_Y", "Biomarker_Y": "Biomarker_X"}
    )
    corr_full = pd.concat([corr_df, mirror_df], ignore_index=True)

    # Add diagonal (correlation = 1)
    diag_rows = [
        {"Biomarker_X": b, "Biomarker_Y": b, "Correlation": 1.0} for b in biomarkers
    ]
    corr_full = pd.concat([corr_full, pd.DataFrame(diag_rows)], ignore_index=True)

    # --- Altair charts: heatmap + scatter linked by selection ---
    selection = alt.selection_point(
        fields=["Biomarker_X", "Biomarker_Y"],
        empty="none",  # no scatter until a cell is clicked
        name="pair",
    )

    heatmap = (
        alt.Chart(corr_full)
        .mark_rect()
        .encode(
            x=alt.X("Biomarker_X:N", title="", sort=biomarkers),
            y=alt.Y("Biomarker_Y:N", title="", sort=biomarkers),
            color=alt.Color(
                "Correlation:Q",
                title="r",
                scale=alt.Scale(domain=[-1, 0, 1], range=["#b2182b", "#f7f7f7", "#2166ac"]),
            ),
            opacity=alt.condition(selection, alt.value(1.0), alt.value(0.7)),
            tooltip=[
                alt.Tooltip("Biomarker_X:N", title="X biomarker"),
                alt.Tooltip("Biomarker_Y:N", title="Y biomarker"),
                alt.Tooltip("Correlation:Q", title="r", format=".2f"),
            ],
        )
        .add_params(selection)
        .properties(height=350)
    )

    st.markdown("**Correlation heatmap (click a cell to see the scatterplot below)**")
    st.altair_chart(heatmap, use_container_width=True)

    # Scatter: filter by selected pair
    scatter = (
        alt.Chart(long_df)
        .transform_filter(selection)
        .mark_circle(size=40, opacity=0.5)
        .encode(
            x=alt.X("x_value:Q", title="Biomarker X value"),
            y=alt.Y("y_value:Q", title="Biomarker Y value"),
            tooltip=[
                alt.Tooltip("Biomarker_X:N", title="X biomarker"),
                alt.Tooltip("Biomarker_Y:N", title="Y biomarker"),
                alt.Tooltip("x_value:Q", title="X value", format=".2f"),
                alt.Tooltip("y_value:Q", title="Y value", format=".2f"),
            ],
        )
        .properties(height=300)
    )

    st.markdown("**Selected pair: detailed scatterplot**")
    st.altair_chart(scatter, use_container_width=True)



# ---- First Graph section (Shu) ----
age_stratified_section(filtered)

# ---- Second Graph section (Shu) ----
correlation_matrix_section(filtered)




# ---------------
# You's app code
# ---------------

# -----------------------------------------------
# 3. Metabolic markers across comorbidity levels
# -----------------------------------------------

st.markdown("#### 3. Metabolic marker levels in selected disease groups")

selected_biomarker = st.selectbox(
    "Select a biomarker to compare",
    options=METABOLIC_COLS,
    index=0,
    format_func=nice_label,
    key="task3_biomarker"
)

selected_outcomes = st.multiselect(
    "Select one or more conditions to compare",
    options=OUTCOME_COLS,
    default=[OUTCOME_COLS[0]],
    format_func=nice_label,
)

if selected_outcomes:
    long_frames = []
    for cond in selected_outcomes:
        tmp_yes = filtered[
            filtered[cond].astype(str).str.upper() == "YES"
        ][[selected_biomarker, "Gender"]].copy()
        tmp_yes["Condition"] = nice_label(cond)
        tmp_yes["Status"] = "With condition"
        
        tmp_no = filtered[
            filtered[cond].astype(str).str.upper() == "NO"
        ][[selected_biomarker, "Gender"]].copy()
        tmp_no["Condition"] = nice_label(cond)
        tmp_no["Status"] = "Without condition"
        
        long_frames.extend([tmp_yes, tmp_no])

    if long_frames:
        subset_long = pd.concat(long_frames, ignore_index=True)

        if subset_long.empty:
            st.warning("No participants available under the current filters.")
        else:
            counts = subset_long.groupby(["Condition", "Status"]).size().reset_index(name="n")
            
            st.caption("**Sample sizes and summary statistics:**")
            for cond in selected_outcomes:
                cond_nice = nice_label(cond)
                with_cond = subset_long[(subset_long["Condition"] == cond_nice) & 
                                       (subset_long["Status"] == "With condition")][selected_biomarker]
                without_cond = subset_long[(subset_long["Condition"] == cond_nice) & 
                                          (subset_long["Status"] == "Without condition")][selected_biomarker]
                
                if len(with_cond) > 0 and len(without_cond) > 0:
                    st.caption(
                        f"**{cond_nice}:** "
                        f"With (n={len(with_cond)}, median={with_cond.median():.1f}) vs "
                        f"Without (n={len(without_cond)}, median={without_cond.median():.1f})"
                    )

            density_chart = (
                alt.Chart(subset_long)
                .transform_density(
                    density=selected_biomarker,
                    groupby=["Condition", "Status"],
                    as_=[selected_biomarker, "density"]
                )
                .mark_area(opacity=0.5)
                .encode(
                    x=alt.X(f"{selected_biomarker}:Q", title=nice_label(selected_biomarker)),
                    y=alt.Y("density:Q", title="Density"),
                    color=alt.Color(
                        "Status:N",
                        title="Disease Status",
                        scale=alt.Scale(
                            domain=["With condition", "Without condition"],
                            range=["#e74c3c", "#3498db"]  
                        ),
                        legend=alt.Legend(orient="top")
                    ),
                    tooltip=[
                        alt.Tooltip("Condition:N", title="Condition"),
                        alt.Tooltip("Status:N", title="Status")
                    ]
                )
                .properties(height=350)
                .facet(
                    column=alt.Column(
                        "Condition:N",
                        title="Condition",
                        header=alt.Header(labelAngle=0, labelAlign="center")
                    )
                )
                .resolve_scale(y="independent")
            )

            st.altair_chart(density_chart, use_container_width=True)
            
    else:
        st.warning("No data available for the selected conditions.")
else:
    st.caption("Select one or more conditions above to compare their metabolic marker distributions.")


# -----------------------------------------------
# 4. Biomarker progression across the lifespan
# -----------------------------------------------

st.markdown("#### 4. Biomarker trends across age with clinical thresholds")

selected_biomarker_age = st.selectbox(
    "Select a biomarker to view age trends",
    options=METABOLIC_COLS,
    index=0,
    format_func=nice_label,
    key="task4_biomarker"
)

# Default clinical thresholds for different biomarkers
DEFAULT_THRESHOLDS = {
    "LDL_Cholesterol": [
        {"value": 100, "label": "Optimal", "color": "#2ecc71"},
        {"value": 130, "label": "Borderline High", "color": "#f39c12"},
        {"value": 160, "label": "High", "color": "#e74c3c"}
    ],
    "HDL_Cholesterol": [
        {"value": 40, "label": "Low (men)", "color": "#e74c3c"},
        {"value": 50, "label": "Low (women)", "color": "#f39c12"},
        {"value": 60, "label": "Protective", "color": "#2ecc71"}
    ],
    "Systolic_BP_Average": [
        {"value": 120, "label": "Normal", "color": "#2ecc71"},
        {"value": 130, "label": "Elevated", "color": "#f39c12"},
        {"value": 140, "label": "Stage 1 HTN", "color": "#e67e22"},
        {"value": 180, "label": "Stage 2 HTN", "color": "#e74c3c"}
    ],
    "Diastolic_BP_Average": [
        {"value": 80, "label": "Normal", "color": "#2ecc71"},
        {"value": 90, "label": "Stage 1 HTN", "color": "#e67e22"},
        {"value": 120, "label": "Stage 2 HTN", "color": "#e74c3c"}
    ],
    "Serum_Glucose": [
        {"value": 100, "label": "Normal", "color": "#2ecc71"},
        {"value": 126, "label": "Prediabetes", "color": "#f39c12"},
        {"value": 200, "label": "Diabetes", "color": "#e74c3c"}
    ],
    "Glycohemoglobin": [
        {"value": 5.7, "label": "Normal", "color": "#2ecc71"},
        {"value": 6.5, "label": "Prediabetes", "color": "#f39c12"},
        {"value": 10, "label": "Diabetes", "color": "#e74c3c"}
    ],
    "BMI": [
        {"value": 18.5, "label": "Underweight", "color": "#3498db"},
        {"value": 25, "label": "Normal", "color": "#2ecc71"},
        {"value": 30, "label": "Overweight", "color": "#f39c12"},
        {"value": 40, "label": "Obese", "color": "#e74c3c"}
    ],
    "Waist_Circumference": [
        {"value": 88, "label": "High risk (women)", "color": "#f39c12"},
        {"value": 102, "label": "High risk (men)", "color": "#e74c3c"}
    ],
    "Triglycerides": [
        {"value": 150, "label": "Normal", "color": "#2ecc71"},
        {"value": 200, "label": "Borderline High", "color": "#f39c12"},
        {"value": 500, "label": "High", "color": "#e74c3c"}
    ],
    "Total_Cholesterol": [
        {"value": 200, "label": "Desirable", "color": "#2ecc71"},
        {"value": 240, "label": "Borderline High", "color": "#f39c12"},
        {"value": 300, "label": "High", "color": "#e74c3c"}
    ],
    "Creatinine": [
        {"value": 0.7, "label": "Low normal", "color": "#3498db"},
        {"value": 1.3, "label": "Normal", "color": "#2ecc71"},
        {"value": 1.5, "label": "Elevated", "color": "#f39c12"}
    ],
    "Uric_Acid": [
        {"value": 4, "label": "Normal lower", "color": "#2ecc71"},
        {"value": 7, "label": "Normal upper (men)", "color": "#f39c12"},
        {"value": 6, "label": "Normal upper (women)", "color": "#f39c12"}
    ],
    "Hemoglobin": [
        {"value": 12, "label": "Low (women)", "color": "#e74c3c"},
        {"value": 13.5, "label": "Low (men)", "color": "#f39c12"},
        {"value": 17.5, "label": "Normal upper", "color": "#2ecc71"}
    ],
    "Hematocrit": [
        {"value": 36, "label": "Low (women)", "color": "#e74c3c"},
        {"value": 39, "label": "Low (men)", "color": "#f39c12"},
        {"value": 50, "label": "Normal upper", "color": "#2ecc71"}
    ],
    "WBC_Count": [
        {"value": 4.5, "label": "Low", "color": "#3498db"},
        {"value": 11, "label": "Normal upper", "color": "#2ecc71"},
        {"value": 15, "label": "Elevated", "color": "#e74c3c"}
    ],
    "Platelet_Count": [
        {"value": 150, "label": "Low", "color": "#e74c3c"},
        {"value": 400, "label": "Normal upper", "color": "#2ecc71"},
        {"value": 450, "label": "Elevated", "color": "#f39c12"}
    ]
}

st.markdown("##### Customize clinical thresholds (optional)")
with st.expander("Adjust thresholds for selected biomarker"):
    if selected_biomarker_age in DEFAULT_THRESHOLDS:
        st.caption(f"Adjust thresholds for **{nice_label(selected_biomarker_age)}**")
        
        # Allow users to modify thresholds
        custom_thresholds = []
        default_thresholds = DEFAULT_THRESHOLDS[selected_biomarker_age]
        
        num_thresholds = st.number_input(
            "Number of thresholds",
            min_value=0,
            max_value=6,
            value=len(default_thresholds),
            key="num_thresholds"
        )
        
        if num_thresholds > 0:
            cols = st.columns(3)
            for i in range(num_thresholds):
                default = default_thresholds[i] if i < len(default_thresholds) else {"value": 0, "label": f"Threshold {i+1}", "color": "#95a5a6"}
                
                with cols[0]:
                    value = st.number_input(
                        f"Value {i+1}",
                        value=float(default["value"]),
                        key=f"threshold_val_{i}",
                        format="%.2f"
                    )
                with cols[1]:
                    label = st.text_input(
                        f"Label {i+1}",
                        value=default["label"],
                        key=f"threshold_label_{i}"
                    )
                with cols[2]:
                    color = st.color_picker(
                        f"Color {i+1}",
                        value=default["color"],
                        key=f"threshold_color_{i}"
                    )
                
                custom_thresholds.append({"value": value, "label": label, "color": color})
        
        CLINICAL_THRESHOLDS = {selected_biomarker_age: custom_thresholds} if custom_thresholds else {}
    else:
        st.info(f"No default thresholds defined for {nice_label(selected_biomarker_age)}. You can add custom thresholds below.")
        
        num_thresholds = st.number_input(
            "Number of thresholds to add",
            min_value=0,
            max_value=6,
            value=0,
            key="num_custom_thresholds"
        )
        
        custom_thresholds = []
        if num_thresholds > 0:
            cols = st.columns(3)
            for i in range(num_thresholds):
                with cols[0]:
                    value = st.number_input(
                        f"Value {i+1}",
                        value=0.0,
                        key=f"custom_threshold_val_{i}",
                        format="%.2f"
                    )
                with cols[1]:
                    label = st.text_input(
                        f"Label {i+1}",
                        value=f"Threshold {i+1}",
                        key=f"custom_threshold_label_{i}"
                    )
                with cols[2]:
                    color = st.color_picker(
                        f"Color {i+1}",
                        value="#95a5a6",
                        key=f"custom_threshold_color_{i}"
                    )
                
                custom_thresholds.append({"value": value, "label": label, "color": color})
        
        CLINICAL_THRESHOLDS = {selected_biomarker_age: custom_thresholds} if custom_thresholds else {}

# If no custom thresholds were set, use defaults
if selected_biomarker_age not in CLINICAL_THRESHOLDS and selected_biomarker_age in DEFAULT_THRESHOLDS:
    CLINICAL_THRESHOLDS = {selected_biomarker_age: DEFAULT_THRESHOLDS[selected_biomarker_age]}



age_data = filtered[["Age", selected_biomarker_age, "Gender"]].dropna()

if age_data.empty:
    st.warning("No data available for the selected biomarker and current filters.")
else:
    st.caption(f"**Age trend analysis for {nice_label(selected_biomarker_age)}**")
    total_n = len(age_data)
    male_n = len(age_data[age_data["Gender"] == "Male"])
    female_n = len(age_data[age_data["Gender"] == "Female"])
    st.caption(f"Total participants: {total_n} (Male: {male_n}, Female: {female_n})")
    
    charts = []
    
    for gender in ["Male", "Female"]:
        gender_data = age_data[age_data["Gender"] == gender].copy()
        
        if gender_data.empty:
            continue
        
        gender_data = gender_data.sort_values("Age")
        
       
        scatter = alt.Chart(gender_data).mark_circle(
            opacity=0.1,
            size=20
        ).encode(
            x=alt.X("Age:Q", title="Age (years)", scale=alt.Scale(domain=[20, 80])),
            y=alt.Y(f"{selected_biomarker_age}:Q", title=nice_label(selected_biomarker_age))
        )
        
        median_line = alt.Chart(gender_data).mark_line(
            size=3
        ).transform_loess(
            "Age",
            selected_biomarker_age,
            bandwidth=0.3  
        ).encode(
            x=alt.X("Age:Q"),
            y=alt.Y(f"{selected_biomarker_age}:Q"),
            color=alt.value("#ff69b4" if gender == "Female" else "#1f77b4")
        )
        
        
        band_data = gender_data.copy()
        
        p10_line = alt.Chart(gender_data).mark_line(
            opacity=0
        ).transform_quantile_regression(
            "Age",
            selected_biomarker_age,
            quantile=0.1
        ).encode(
            x=alt.X("Age:Q"),
            y=alt.Y(f"{selected_biomarker_age}:Q")
        )
        
        p90_line = alt.Chart(gender_data).mark_line(
            opacity=0
        ).transform_quantile_regression(
            "Age",
            selected_biomarker_age,
            quantile=0.9
        ).encode(
            x=alt.X("Age:Q"),
            y=alt.Y(f"{selected_biomarker_age}:Q")
        )
        
        window_size = max(20, len(gender_data) // 20)  
        gender_data_sorted = gender_data.sort_values("Age")
        gender_data_sorted["P10"] = gender_data_sorted[selected_biomarker_age].rolling(
            window=window_size, center=True, min_periods=5
        ).quantile(0.1)
        gender_data_sorted["P90"] = gender_data_sorted[selected_biomarker_age].rolling(
            window=window_size, center=True, min_periods=5
        ).quantile(0.9)
        
        band = alt.Chart(gender_data_sorted).mark_area(
            opacity=0.2
        ).encode(
            x=alt.X("Age:Q"),
            y=alt.Y("P10:Q", title=nice_label(selected_biomarker_age)),
            y2=alt.Y2("P90:Q"),
            color=alt.value("#ff69b4" if gender == "Female" else "#1f77b4")
        )
        
        threshold_layers = []
        if selected_biomarker_age in CLINICAL_THRESHOLDS:
            for threshold in CLINICAL_THRESHOLDS[selected_biomarker_age]:
                threshold_line = alt.Chart(pd.DataFrame({
                    "threshold": [threshold["value"]],
                    "label": [threshold["label"]]
                })).mark_rule(
                    strokeDash=[5, 5],
                    opacity=0.7,
                    size=2
                ).encode(
                    y=alt.Y("threshold:Q"),
                    color=alt.value(threshold["color"]),
                    tooltip=alt.Tooltip("label:N", title="Clinical threshold")
                )
                threshold_layers.append(threshold_line)
        
        chart = band + median_line
        for tl in threshold_layers:
            chart = chart + tl
        
        chart = chart.properties(
            width=350,
            height=400,
            title=f"{gender}"
        )
        
        charts.append(chart)
    
    if len(charts) == 2:
        combined_chart = alt.hconcat(*charts).resolve_scale(y="shared")
        st.altair_chart(combined_chart, use_container_width=True)
    elif len(charts) == 1:
        st.altair_chart(charts[0], use_container_width=True)
    
    if selected_biomarker_age in CLINICAL_THRESHOLDS:
        st.caption("**Clinical thresholds:**")
        threshold_text = " | ".join([
            f"{t['label']}: {t['value']}" for t in CLINICAL_THRESHOLDS[selected_biomarker_age]
        ])
        st.caption(threshold_text)
        
        st.caption("*Thresholds based on clinical guidelines (AHA, ADA, ACC, CDC, WHO). Individual risk assessment should consider multiple factors.*")




# -----------------------------
# Kathy's original app code
# -----------------------------


# -----------------------------
# 1. Disease prevalence summary
# -----------------------------
st.subheader("3. Comorbidity overview under current filters")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "Any comorbidity (≥1 condition)",
        f"{(filtered['Any_Comorbidity'] == 'Yes').mean() * 100:.1f} %",
    )
with col2:
    st.metric(
        "Mean comorbidity count",
        f"{filtered['Comorbidity_Count'].mean():.2f}",
    )
with col3:
    st.metric(
        "Max comorbidity count",
        int(filtered["Comorbidity_Count"].max()),
    )

prev_df = prevalence_table(filtered)

prev_chart = (
    alt.Chart(prev_df)
    .mark_bar()
    .encode(
        x=alt.X("Prevalence (%):Q", title="Prevalence (%)"),
        y=alt.Y("Outcome:N", sort="-x", title=""),
        tooltip=["Outcome", "Prevalence (%)"],
    )
    .properties(height=300)
)

st.altair_chart(prev_chart, use_container_width=True)


## -----------------------------
# 2. Metabolic markers across comorbidity levels
# -----------------------------

st.subheader("4. Metabolic markers across comorbidity levels")

cm_y_var = st.selectbox(
    "Metabolic variable (y-axis, by comorbidity count)",
    options=METABOLIC_COLS,
    format_func=nice_label,
    index=METABOLIC_COLS.index("BMI") if "BMI" in METABOLIC_COLS else 0,
)

genders_in_data = (
    filtered["Gender"]
    .dropna()
    .astype(str)
    .unique()
)

# Single plot: boxes + points, grouped and colored by gender
base = (
    alt.Chart(filtered)
    .encode(
        x=alt.X(
            "Comorbidity_Count:O",
            title="Number of comorbid conditions"
        ),
        # horizontally dodge male/female within each count
        xOffset=alt.XOffset("Gender:N"),
        y=alt.Y(
            cm_y_var,
            title=nice_label(cm_y_var)
        ),
        color=alt.Color(
            "Gender:N",
            title="Gender",
            scale=alt.Scale(
                domain=["Female", "Male"],           # adjust if your labels differ
                range=["#ff69b4", "#1f77b4"]         # pink for female, blue for male
            ),
        ),
        tooltip=["Gender", "Comorbidity_Count:O", cm_y_var],
    )
)

box = base.mark_boxplot()
points = base.mark_circle(size=20, opacity=0.3)

metab_chart = (box + points).properties(height=350)

st.altair_chart(metab_chart, use_container_width=True)




# --- Additional view: compare metabolic marker across selected diseases ---
st.markdown("#### Metabolic marker levels in selected disease groups")

selected_outcomes = st.multiselect(
    "Select one or more conditions to compare",
    options=OUTCOME_COLS,
    format_func=nice_label,
)

if selected_outcomes:
    # Build a long-format dataframe: one row per person-condition pair
    long_frames = []
    for cond in selected_outcomes:
        # keep only participants with this condition = Yes
        tmp = filtered[
            filtered[cond].astype(str).str.upper() == "YES"
        ][[cm_y_var, "Gender"]].copy()
        tmp["Condition"] = nice_label(cond)
        long_frames.append(tmp)

    if long_frames:
        subset_long = pd.concat(long_frames, ignore_index=True)

        if subset_long.empty:
            st.warning("No participants have any of the selected conditions under the current filters.")
        else:
            # Show sample sizes per condition
            counts = subset_long.groupby(["Condition", "Gender"]).size().reset_index(name="n")
            count_str = "; ".join(
                [f"{row['Condition']} ({row['Gender']}): n={row['n']}" for _, row in counts.iterrows()]
            )
            st.caption(f"Number of participants per condition (showing those with the condition): {count_str}")

            cond_box = (
                alt.Chart(subset_long)
                .mark_boxplot()
                .encode(
                    x=alt.X("Condition:N", title="Condition"),
                    # dodge male/female within each condition
                    xOffset=alt.XOffset("Gender:N"),
                    y=alt.Y(cm_y_var, title=nice_label(cm_y_var)),
                    color=alt.Color(
                        "Gender:N",
                        title="Gender",
                        scale=alt.Scale(
                            domain=["Female", "Male"],
                            range=["#ff69b4", "#1f77b4"]
                        ),
                    ),
                    tooltip=["Gender:N", "Condition:N", cm_y_var]
                )
                .properties(height=350)
            )


            st.altair_chart(cond_box, use_container_width=True)
    else:
        st.warning("No data available for the selected conditions.")
else:
    st.caption("Select one or more conditions above to compare their metabolic marker distributions.")


# -----------------------------
# 3. Comorbidity / disease prevalence by lifestyle level
# -----------------------------
st.subheader("5. Comorbidity / disease prevalence by lifestyle level")

life_var = st.selectbox(
    "Lifestyle variable to group by",
    options=LIFESTYLE_COLS,
    format_func=nice_label,
    index=LIFESTYLE_COLS.index("Physical_Activity_Equivalent_Min")
      if "Physical_Activity_Equivalent_Min" in LIFESTYLE_COLS else 0,
)

# choose one disease
outcome_prev = st.selectbox(
    "Outcome whose prevalence to plot",
    options=["Any_Comorbidity"] + OUTCOME_COLS,
    format_func=lambda v: "Any comorbidity (≥1)" if v == "Any_Comorbidity" else nice_label(v),
)

tmp = filtered.copy()

if outcome_prev == "Any_Comorbidity":
    tmp["Outcome_Num"] = (tmp["Any_Comorbidity"] == "Yes").astype(int)
    outcome_title = "Any comorbidity prevalence"
else:
    tmp["Outcome_Num"] = (tmp[outcome_prev].astype(str).str.upper() == "YES").astype(int)
    outcome_title = f"{nice_label(outcome_prev)} prevalence"

prev_chart = (
    alt.Chart(tmp)
    .transform_bin(
        bin=alt.Bin(maxbins=15),
        field=life_var,
        as_="life_bin"
    )
    .transform_aggregate(
        prevalence="mean(Outcome_Num)",
        groupby=["life_bin", "Gender"]
    )
    .mark_line(point=True)
    .encode(
        x=alt.X(
            "life_bin:Q",
            title=nice_label(life_var)
        ),
        y=alt.Y(
            "prevalence:Q",
            title=outcome_title,
            axis=alt.Axis(format=".0%")
        ),
        color=alt.Color(
            "Gender:N",
            title="Gender",
            scale=alt.Scale(
                domain=["Female", "Male"],
                range=["#ff69b4", "#1f77b4"]
            ),
        ),
        order=alt.Order("life_bin:Q", sort="ascending"),
        tooltip=[
            alt.Tooltip("Gender:N", title="Gender"),
            alt.Tooltip("life_bin:Q", title=nice_label(life_var)),
            alt.Tooltip("prevalence:Q", format=".1%", title="Prevalence"),
        ]
    )
    .properties(height=300)
)

st.altair_chart(prev_chart, use_container_width=True)
