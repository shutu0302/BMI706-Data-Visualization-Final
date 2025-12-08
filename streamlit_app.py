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
# Main app
# -----------------------------


# -----------------------------
# New from Shu: Age-stratified risk-factor section
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
    st.subheader("Age-stratified distribution of risk factors")

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
# Main app
# -----------------------------
df = load_data(DATA_PATH)

st.title("Lifestyle & Comorbidity Explorer")
st.markdown(
    """
This dashboard lets you interactively explore how **lifestyle behaviours**  
(smoking, sleep, physical activity, alcohol) and **demographics** relate to:

- Metabolic and cardiometabolic markers (BMI, lipids, glucose, blood pressure, etc.)
- Prevalence and burden of chronic conditions (heart attack, diabetes, stroke, COPD, etc.)
"""
)

filtered = apply_filters(df)
st.markdown(f"### Current selection: {len(filtered):,} participants")

if filtered.empty:
    st.warning("No participants match the current filter settings. Try relaxing your filters.")
    st.stop()

# ---- First Graph section (Shu) ----
age_stratified_section(filtered)



# -----------------------------
# Kathy's original app code
# -----------------------------


# -----------------------------
# 1. Disease prevalence summary
# -----------------------------
st.subheader("Comorbidity overview under current filters")

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

st.subheader("Metabolic markers across comorbidity levels")

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
st.subheader("Comorbidity / disease prevalence by lifestyle level")

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
