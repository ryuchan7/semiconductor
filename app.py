import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Semiconductor Process Intelligence",
    page_icon="⚙️",
    layout="wide"
)


# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_models():

    rf_model = joblib.load(
        "models/random_forest.pkl"
    )

    rf_imputer = joblib.load(
        "models/rf_imputer.pkl"
    )

    iso_model = joblib.load(
        "models/isolation_forest.pkl"
    )

    anomaly_imputer = joblib.load(
        "models/anomaly_imputer.pkl"
    )

    feature_names = joblib.load(
        "models/feature_names.pkl"
    )

    thresholds = joblib.load(
        "models/thresholds.pkl"
    )

    explainer = shap.TreeExplainer(
        rf_model
    )

    return (
        rf_model,
        rf_imputer,
        iso_model,
        anomaly_imputer,
        feature_names,
        thresholds,
        explainer
    )


# ==========================================================
# LOAD DEMO DATA
# ==========================================================

@st.cache_data
def load_demo_data():

    return pd.read_csv(
        "data/demo_samples.csv"
    )


(
    rf_model,
    rf_imputer,
    iso_model,
    anomaly_imputer,
    feature_names,
    thresholds,
    explainer
) = load_models()


demo_data = load_demo_data()


# ==========================================================
# THRESHOLDS
# ==========================================================

rf_threshold = thresholds[
    "rf_threshold"
]

anomaly_threshold = thresholds[
    "anomaly_threshold"
]


# ==========================================================
# RANDOM FOREST FAIL CLASS INDEX
# ==========================================================

fail_class_index = list(
    rf_model.classes_
).index(1)


# ==========================================================
# TITLE
# ==========================================================

st.title(
    "Semiconductor Process Intelligence"
)

st.caption(
    "AI-based yield failure prediction, "
    "process anomaly detection and explainable AI"
)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header(
    "Analysis Mode"
)

analysis_mode = st.sidebar.radio(
    "Choose data source",
    [
        "Demo Samples",
        "Upload CSV"
    ]
)


st.sidebar.divider()

st.sidebar.subheader(
    "AI Models"
)

st.sidebar.write(
    "🌲 Random Forest"
)

st.sidebar.write(
    "🔎 Isolation Forest"
)

st.sidebar.write(
    "🧠 SHAP Explainability"
)


st.sidebar.divider()

st.sidebar.caption(
    "Semiconductor Process Intelligence Prototype"
)


# ==========================================================
# FUNCTION:
# ANALYZE ONE SAMPLE
# ==========================================================

def analyze_single_sample(X_sample):

    # ------------------------------------------------------
    # RANDOM FOREST
    # ------------------------------------------------------

    X_rf = rf_imputer.transform(
        X_sample
    )

    fail_probability = (
        rf_model.predict_proba(
            X_rf
        )[:, fail_class_index][0]
    )

    rf_prediction = (
        1
        if fail_probability >= rf_threshold
        else -1
    )


    # ------------------------------------------------------
    # ISOLATION FOREST
    # ------------------------------------------------------

    X_iso = anomaly_imputer.transform(
        X_sample
    )

    anomaly_score = (
        -iso_model.score_samples(
            X_iso
        )[0]
    )

    anomaly_prediction = (
        anomaly_score >= anomaly_threshold
    )


    # ------------------------------------------------------
    # SHAP
    # ------------------------------------------------------

    sample_shap_values = (
        explainer.shap_values(
            X_rf
        )
    )


    # ------------------------------------------------------
    # HANDLE SHAP OUTPUT FORMAT
    # ------------------------------------------------------

    if isinstance(
        sample_shap_values,
        list
    ):

        shap_fail_sample = np.asarray(
            sample_shap_values[
                fail_class_index
            ]
        )[0]

    else:

        sample_shap_array = np.asarray(
            sample_shap_values
        )

        # Example:
        # (1, 446, 2)

        if sample_shap_array.ndim == 3:

            shap_fail_sample = (
                sample_shap_array[
                    0,
                    :,
                    fail_class_index
                ]
            )

        # Example:
        # (1, 446)

        elif sample_shap_array.ndim == 2:

            shap_fail_sample = (
                sample_shap_array[0]
            )

        else:

            raise ValueError(
                "Unexpected SHAP output format."
            )


    # ------------------------------------------------------
    # SHAP TABLE
    # ------------------------------------------------------

    shap_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Feature Value": X_rf[0],
            "SHAP Value": shap_fail_sample
        }
    )

    shap_df[
        "Importance"
    ] = np.abs(
        shap_df[
            "SHAP Value"
        ]
    )

    shap_df[
        "Direction"
    ] = np.where(
        shap_df[
            "SHAP Value"
        ] > 0,
        "Toward FAIL",
        "Toward PASS"
    )

    shap_df = shap_df.sort_values(
        by="Importance",
        ascending=False
    )


    return (
        fail_probability,
        rf_prediction,
        anomaly_score,
        anomaly_prediction,
        shap_df
    )


# ==========================================================
# FUNCTION:
# DISPLAY DETAILED SAMPLE ANALYSIS
# ==========================================================

def display_single_sample_analysis(
    X_sample,
    actual_class=None
):

    (
        fail_probability,
        rf_prediction,
        anomaly_score,
        anomaly_prediction,
        shap_df
    ) = analyze_single_sample(
        X_sample
    )


    # ======================================================
    # MAIN METRICS
    # ======================================================

    st.divider()


    if actual_class is not None:

        col1, col2, col3, col4 = (
            st.columns(4)
        )

    else:

        col1, col2, col3 = (
            st.columns(3)
        )


    with col1:

        st.metric(
            "Yield Failure Risk",
            f"{fail_probability * 100:.1f}%"
        )


    with col2:

        st.metric(
            "Anomaly Score",
            f"{anomaly_score:.3f}"
        )


    with col3:

        st.metric(
            "AI Prediction",
            (
                "FAIL"
                if rf_prediction == 1
                else "PASS"
            )
        )


    if actual_class is not None:

        with col4:

            st.metric(
                "Actual Result",
                (
                    "FAIL"
                    if actual_class == 1
                    else "PASS"
                )
            )


    # ======================================================
    # AI ASSESSMENT
    # ======================================================

    st.header(
        "AI Assessment"
    )


    if rf_prediction == 1:

        st.error(
            "⚠ High Yield Failure Risk"
        )

    else:

        st.success(
            "✓ Low Yield Failure Risk"
        )


    if anomaly_prediction:

        st.warning(
            "⚠ Abnormal Process Behaviour Detected"
        )

    else:

        st.success(
            "✓ Process Behaviour Appears Normal"
        )


    if actual_class is not None:

        if rf_prediction == actual_class:

            st.success(
                "✓ AI prediction matches the actual result"
            )

        else:

            st.error(
                "✗ AI prediction does not match the actual result"
            )


    # ======================================================
    # PROCESS STATUS SUMMARY
    # ======================================================

    st.subheader(
        "Process Status Summary"
    )


    status_col1, status_col2 = (
        st.columns(2)
    )


    with status_col1:

        st.write(
            "**Yield Risk Assessment**"
        )

        st.write(
            f"Failure probability: "
            f"**{fail_probability * 100:.2f}%**"
        )

        st.write(
            f"Decision threshold: "
            f"**{rf_threshold * 100:.2f}%**"
        )


    with status_col2:

        st.write(
            "**Process Anomaly Assessment**"
        )

        st.write(
            f"Anomaly score: "
            f"**{anomaly_score:.4f}**"
        )

        st.write(
            f"Anomaly threshold: "
            f"**{anomaly_threshold:.6f}**"
        )


    # ======================================================
    # COMBINED AI STATUS
    # ======================================================

    st.subheader(
        "Combined AI Status"
    )


    if (
        rf_prediction == 1
        and anomaly_prediction
    ):

        st.error(
            "🚨 HIGH RISK — Failure prediction and "
            "anomaly detection both raised an alert."
        )

    elif (
        rf_prediction == 1
        or anomaly_prediction
    ):

        st.warning(
            "⚠ WARNING — One AI model detected "
            "potentially suspicious process behaviour."
        )

    else:

        st.success(
            "✓ NORMAL — No major process alert detected."
        )


    # ======================================================
    # EXPLAINABLE AI
    # ======================================================

    st.divider()

    st.header(
        "AI Explainability"
    )

    st.write(
        "SHAP explains which semiconductor process "
        "variables had the strongest influence on the "
        "Random Forest prediction for this sample."
    )


    # ------------------------------------------------------
    # FACTORS TOWARD FAIL
    # ------------------------------------------------------

    toward_fail = (
        shap_df[
            shap_df[
                "SHAP Value"
            ] > 0
        ]
        .sort_values(
            by="SHAP Value",
            ascending=False
        )
        .head(10)
    )


    # ------------------------------------------------------
    # FACTORS TOWARD PASS
    # ------------------------------------------------------

    toward_pass = (
        shap_df[
            shap_df[
                "SHAP Value"
            ] < 0
        ]
        .sort_values(
            by="SHAP Value",
            ascending=True
        )
        .head(10)
    )


    col_fail, col_pass = (
        st.columns(2)
    )


    with col_fail:

        st.subheader(
            "⚠ Factors Toward FAIL"
        )

        if len(toward_fail) > 0:

            st.dataframe(
                toward_fail[
                    [
                        "Feature",
                        "Feature Value",
                        "SHAP Value"
                    ]
                ],
                hide_index=True,
                use_container_width=True
            )

        else:

            st.info(
                "No variables are pushing "
                "the model toward FAIL."
            )


    with col_pass:

        st.subheader(
            "✓ Factors Toward PASS"
        )

        if len(toward_pass) > 0:

            st.dataframe(
                toward_pass[
                    [
                        "Feature",
                        "Feature Value",
                        "SHAP Value"
                    ]
                ],
                hide_index=True,
                use_container_width=True
            )

        else:

            st.info(
                "No variables are pushing "
                "the model toward PASS."
            )


    # ======================================================
    # SHAP BAR CHART
    # ======================================================

    st.subheader(
        "Top Process Variables Influencing Prediction"
    )


    top_shap = (
        shap_df
        .head(15)
        .sort_values(
            by="SHAP Value"
        )
    )


    fig_shap, ax_shap = plt.subplots(
        figsize=(10, 7)
    )


    ax_shap.barh(
        top_shap[
            "Feature"
        ],
        top_shap[
            "SHAP Value"
        ]
    )


    ax_shap.axvline(
        x=0,
        linewidth=1
    )


    ax_shap.set_xlabel(
        "SHAP Value"
    )

    ax_shap.set_ylabel(
        "Process Variable"
    )

    ax_shap.set_title(
        "Top Variables Influencing This Prediction"
    )


    plt.tight_layout()


    st.pyplot(
        fig_shap,
        clear_figure=True
    )


    # ======================================================
    # TOP 20 SHAP TABLE
    # ======================================================

    with st.expander(
        "View Top 20 Most Influential Variables"
    ):

        st.dataframe(
            shap_df[
                [
                    "Feature",
                    "Feature Value",
                    "SHAP Value",
                    "Importance",
                    "Direction"
                ]
            ].head(20),

            hide_index=True,
            use_container_width=True
        )


# ==========================================================
# MODE 1:
# DEMO SAMPLES
# ==========================================================

if analysis_mode == "Demo Samples":

    st.header(
        "Demo Semiconductor Samples"
    )


    sample_number = st.selectbox(
        "Select semiconductor process sample",
        range(
            len(
                demo_data
            )
        )
    )


    sample = demo_data.iloc[
        [sample_number]
    ]


    actual_class = int(
        sample[
            "Actual_Class"
        ].iloc[0]
    )


    X_sample = sample[
        feature_names
    ]


    display_single_sample_analysis(
        X_sample,
        actual_class
    )


# ==========================================================
# MODE 2:
# CSV UPLOAD
# ==========================================================

else:

    st.header(
        "Upload Semiconductor Process Data"
    )


    st.write(
        "Upload a CSV file containing semiconductor "
        "process measurements for batch AI analysis."
    )


    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=[
            "csv"
        ]
    )


    # ======================================================
    # WAIT FOR FILE
    # ======================================================

    if uploaded_file is None:

        st.info(
            "Upload a CSV file to begin analysis."
        )

        st.stop()


    # ======================================================
    # READ CSV
    # ======================================================

    try:

        uploaded_data = pd.read_csv(
            uploaded_file
        )

    except Exception as error:

        st.error(
            f"Unable to read CSV file: {error}"
        )

        st.stop()


    st.success(
        f"CSV loaded successfully: "
        f"{uploaded_data.shape[0]} samples, "
        f"{uploaded_data.shape[1]} columns"
    )


    # ======================================================
    # VALIDATE FEATURES
    # ======================================================

    missing_features = [
        feature

        for feature in feature_names

        if feature
        not in uploaded_data.columns
    ]


    if len(
        missing_features
    ) > 0:

        st.error(
            f"The CSV is missing "
            f"{len(missing_features)} required "
            f"process variables."
        )


        with st.expander(
            "View Missing Variables"
        ):

            st.write(
                missing_features
            )


        st.stop()


    st.success(
        "✓ All required process variables are available."
    )


    # ======================================================
    # PREPARE UPLOADED DATA
    # ======================================================

    X_uploaded = uploaded_data[
        feature_names
    ].copy()


    for column in feature_names:

        X_uploaded[
            column
        ] = pd.to_numeric(
            X_uploaded[
                column
            ],
            errors="coerce"
        )


    # ======================================================
    # RANDOM FOREST BATCH ANALYSIS
    # ======================================================

    X_uploaded_rf = (
        rf_imputer.transform(
            X_uploaded
        )
    )


    uploaded_fail_probability = (
        rf_model.predict_proba(
            X_uploaded_rf
        )[:, fail_class_index]
    )


    uploaded_rf_prediction = np.where(
        uploaded_fail_probability
        >= rf_threshold,

        1,

        -1
    )


    # ======================================================
    # ISOLATION FOREST BATCH ANALYSIS
    # ======================================================

    X_uploaded_iso = (
        anomaly_imputer.transform(
            X_uploaded
        )
    )


    uploaded_anomaly_score = (
        -iso_model.score_samples(
            X_uploaded_iso
        )
    )


    uploaded_anomaly_prediction = (
        uploaded_anomaly_score
        >= anomaly_threshold
    )


    # ======================================================
    # CREATE BATCH RESULTS
    # ======================================================

    batch_results = pd.DataFrame(
        {
            "Sample": range(
                len(
                    uploaded_data
                )
            ),

            "Failure Risk": (
                uploaded_fail_probability
            ),

            "Failure Risk (%)": (
                uploaded_fail_probability
                * 100
            ),

            "AI Prediction": np.where(
                uploaded_rf_prediction
                == 1,

                "FAIL",

                "PASS"
            ),

            "Anomaly Score": (
                uploaded_anomaly_score
            ),

            "Anomaly Status": np.where(
                uploaded_anomaly_prediction,

                "ANOMALY",

                "NORMAL"
            )
        }
    )


    # ======================================================
    # COMBINED ALERT
    # ======================================================

    batch_results[
        "Combined Alert"
    ] = np.where(

        (
            (
                uploaded_rf_prediction
                == 1
            )
            &
            uploaded_anomaly_prediction
        ),

        "HIGH RISK",

        np.where(

            (
                (
                    uploaded_rf_prediction
                    == 1
                )
                |
                uploaded_anomaly_prediction
            ),

            "WARNING",

            "NORMAL"
        )
    )


    # ======================================================
    # ADD ACTUAL RESULT IF AVAILABLE
    # ======================================================

    if (
        "Actual_Class"
        in uploaded_data.columns
    ):

        batch_results[
            "Actual Result"
        ] = np.where(

            uploaded_data[
                "Actual_Class"
            ].to_numpy()
            == 1,

            "FAIL",

            "PASS"
        )


    # ======================================================
    # SUMMARY COUNTS
    # ======================================================

    total_samples = len(
        batch_results
    )


    high_risk_count = (
        batch_results[
            "Combined Alert"
        ]
        == "HIGH RISK"
    ).sum()


    warning_count = (
        batch_results[
            "Combined Alert"
        ]
        == "WARNING"
    ).sum()


    normal_count = (
        batch_results[
            "Combined Alert"
        ]
        == "NORMAL"
    ).sum()


    predicted_fail_count = (
        batch_results[
            "AI Prediction"
        ]
        == "FAIL"
    ).sum()


    anomaly_count = (
        batch_results[
            "Anomaly Status"
        ]
        == "ANOMALY"
    ).sum()


    # ======================================================
    # BATCH ANALYSIS SUMMARY
    # ======================================================

    st.divider()

    st.header(
        "Batch Analysis Summary"
    )


    col1, col2, col3, col4 = (
        st.columns(4)
    )


    col1.metric(
        "Total Samples",
        int(
            total_samples
        )
    )


    col2.metric(
        "High Risk",
        int(
            high_risk_count
        )
    )


    col3.metric(
        "Warning",
        int(
            warning_count
        )
    )


    col4.metric(
        "Normal",
        int(
            normal_count
        )
    )


    col5, col6 = (
        st.columns(2)
    )


    col5.metric(
        "Predicted FAIL",
        int(
            predicted_fail_count
        )
    )


    col6.metric(
        "Anomalies Detected",
        int(
            anomaly_count
        )
    )


    # ======================================================
    # PRODUCTION BATCH OVERVIEW
    # ======================================================

    st.divider()

    st.header(
        "Production Batch Overview"
    )


    normal_percentage = (
        normal_count
        / total_samples
    ) * 100


    warning_percentage = (
        warning_count
        / total_samples
    ) * 100


    high_risk_percentage = (
        high_risk_count
        / total_samples
    ) * 100


    health_col1, health_col2, health_col3 = (
        st.columns(3)
    )


    health_col1.metric(
        "Normal Samples",
        f"{normal_percentage:.1f}%"
    )


    health_col2.metric(
        "Warning Samples",
        f"{warning_percentage:.1f}%"
    )


    health_col3.metric(
        "High Risk Samples",
        f"{high_risk_percentage:.1f}%"
    )


    # ======================================================
    # OVERALL BATCH STATUS
    # ======================================================

    if high_risk_percentage >= 10:

        st.error(
            "🚨 BATCH STATUS: HIGH RISK — "
            "A significant number of samples "
            "require investigation."
        )

    elif high_risk_percentage > 0:

        st.warning(
            "⚠ BATCH STATUS: ATTENTION REQUIRED — "
            "Some high-risk process samples were detected."
        )

    else:

        st.success(
            "✓ BATCH STATUS: NORMAL — "
            "No high-risk process samples were detected."
        )


    # ======================================================
    # ALERT DISTRIBUTION
    # ======================================================

    st.subheader(
        "Process Alert Distribution"
    )


    alert_counts = (
        batch_results[
            "Combined Alert"
        ]
        .value_counts()
        .reindex(
            [
                "NORMAL",
                "WARNING",
                "HIGH RISK"
            ],
            fill_value=0
        )
    )


    fig_alert, ax_alert = plt.subplots(
        figsize=(8, 5)
    )


    ax_alert.bar(
        alert_counts.index,
        alert_counts.values
    )


    ax_alert.set_ylabel(
        "Number of Samples"
    )

    ax_alert.set_xlabel(
        "Process Status"
    )

    ax_alert.set_title(
        "Production Batch Alert Distribution"
    )


    for index, value in enumerate(
        alert_counts.values
    ):

        ax_alert.text(
            index,
            value,
            str(value),
            ha="center",
            va="bottom"
        )


    plt.tight_layout()


    st.pyplot(
        fig_alert,
        clear_figure=True
    )


    # ======================================================
    # FAILURE RISK DISTRIBUTION
    # ======================================================

    st.subheader(
        "Yield Failure Risk Distribution"
    )


    fig_risk, ax_risk = plt.subplots(
        figsize=(9, 5)
    )


    ax_risk.hist(
        batch_results[
            "Failure Risk (%)"
        ],
        bins=25
    )


    ax_risk.axvline(
        rf_threshold * 100,
        linestyle="--",
        label=(
            f"Failure Threshold "
            f"({rf_threshold * 100:.0f}%)"
        )
    )


    ax_risk.set_xlabel(
        "Predicted Failure Risk (%)"
    )

    ax_risk.set_ylabel(
        "Number of Samples"
    )

    ax_risk.set_title(
        "Distribution of Predicted Yield Failure Risk"
    )


    ax_risk.legend()


    plt.tight_layout()


    st.pyplot(
        fig_risk,
        clear_figure=True
    )


    # ======================================================
    # ANOMALY SCORE DISTRIBUTION
    # ======================================================

    st.subheader(
        "Process Anomaly Score Distribution"
    )


    fig_anomaly, ax_anomaly = plt.subplots(
        figsize=(9, 5)
    )


    ax_anomaly.hist(
        batch_results[
            "Anomaly Score"
        ],
        bins=25
    )


    ax_anomaly.axvline(
        anomaly_threshold,
        linestyle="--",
        label=(
            f"Anomaly Threshold "
            f"({anomaly_threshold:.3f})"
        )
    )


    ax_anomaly.set_xlabel(
        "Anomaly Score"
    )

    ax_anomaly.set_ylabel(
        "Number of Samples"
    )

    ax_anomaly.set_title(
        "Distribution of Process Anomaly Scores"
    )


    ax_anomaly.legend()


    plt.tight_layout()


    st.pyplot(
        fig_anomaly,
        clear_figure=True
    )


    # ======================================================
    # FORMAT DISPLAY RESULTS
    # ======================================================

    display_results = (
        batch_results.copy()
    )


    display_results[
        "Failure Risk (%)"
    ] = display_results[
        "Failure Risk (%)"
    ].round(
        2
    )


    display_results[
        "Anomaly Score"
    ] = display_results[
        "Anomaly Score"
    ].round(
        4
    )


    # ======================================================
    # TOP 10 HIGHEST-RISK SAMPLES
    # ======================================================

    st.subheader(
        "Top 10 Highest-Risk Samples"
    )


    top_risk_samples = (
        display_results
        .sort_values(
            by="Failure Risk (%)",
            ascending=False
        )
        .head(10)
    )


    top_risk_columns = [
        "Sample",
        "Failure Risk (%)",
        "AI Prediction",
        "Anomaly Score",
        "Anomaly Status",
        "Combined Alert"
    ]


    if (
        "Actual Result"
        in top_risk_samples.columns
    ):

        top_risk_columns.append(
            "Actual Result"
        )


    st.dataframe(
        top_risk_samples[
            top_risk_columns
        ],
        hide_index=True,
        use_container_width=True
    )


    # ======================================================
    # RESULT COLUMNS
    # ======================================================

    result_columns = [
        "Sample",
        "Failure Risk (%)",
        "AI Prediction",
        "Anomaly Score",
        "Anomaly Status",
        "Combined Alert"
    ]


    if (
        "Actual Result"
        in display_results.columns
    ):

        result_columns.append(
            "Actual Result"
        )


    # ======================================================
    # PROCESS ANALYSIS RESULTS
    # ======================================================

    st.divider()

    st.subheader(
        "Process Analysis Results"
    )


    st.dataframe(
        display_results[
            result_columns
        ],
        use_container_width=True,
        hide_index=True
    )


    # ======================================================
    # PRIORITY REVIEW QUEUE
    # ======================================================

    st.subheader(
        "Priority Review Queue"
    )


    priority_order = {
        "HIGH RISK": 0,
        "WARNING": 1,
        "NORMAL": 2
    }


    sorted_results = (
        display_results
        .assign(
            Priority=display_results[
                "Combined Alert"
            ].map(
                priority_order
            )
        )
        .sort_values(
            [
                "Priority",
                "Failure Risk (%)"
            ],
            ascending=[
                True,
                False
            ]
        )
    )


    st.dataframe(
        sorted_results[
            result_columns
        ],
        use_container_width=True,
        hide_index=True
    )


    # ======================================================
    # DETAILED SAMPLE INVESTIGATION
    # ======================================================

    st.divider()

    st.header(
        "Detailed Sample Investigation"
    )


    st.write(
        "Select a flagged sample to inspect its "
        "failure risk, anomaly score and SHAP explanation."
    )


    review_samples = (
        sorted_results[
            sorted_results[
                "Combined Alert"
            ] != "NORMAL"
        ]
        .copy()
    )


    if len(
        review_samples
    ) == 0:

        review_samples = (
            sorted_results.copy()
        )


    # ------------------------------------------------------
    # HELPER FOR DROPDOWN LABEL
    # ------------------------------------------------------

    def review_sample_label(sample_id):

        row = batch_results.loc[
            sample_id
        ]

        return (
            f"Sample {sample_id} | "
            f"{row['Combined Alert']} | "
            f"Risk: {row['Failure Risk (%)']:.2f}% | "
            f"Anomaly: {row['Anomaly Score']:.4f}"
        )


    # ------------------------------------------------------
    # SAMPLE SELECTOR
    # ------------------------------------------------------

    selected_review_sample = st.selectbox(
        "Select sample for investigation",
        options=review_samples[
            "Sample"
        ].tolist(),
        format_func=review_sample_label
    )


    # ======================================================
    # RETRIEVE SELECTED SAMPLE
    # ======================================================

    selected_uploaded_sample = (
        X_uploaded.iloc[
            [
                selected_review_sample
            ]
        ]
    )


    # ======================================================
    # ACTUAL CLASS IF AVAILABLE
    # ======================================================

    selected_actual_class = None


    if (
        "Actual_Class"
        in uploaded_data.columns
    ):

        actual_value = (
            uploaded_data.iloc[
                selected_review_sample
            ][
                "Actual_Class"
            ]
        )


        if pd.notna(
            actual_value
        ):

            selected_actual_class = int(
                actual_value
            )


    # ======================================================
    # SELECTED ALERT
    # ======================================================

    selected_alert = (
        batch_results.loc[
            selected_review_sample,
            "Combined Alert"
        ]
    )


    if (
        selected_alert
        == "HIGH RISK"
    ):

        st.error(
            "🚨 HIGH RISK — Failure prediction and "
            "anomaly detection both raised an alert."
        )

    elif (
        selected_alert
        == "WARNING"
    ):

        st.warning(
            "⚠ WARNING — At least one AI system "
            "detected suspicious process behaviour."
        )

    else:

        st.success(
            "✓ NORMAL — No major process alert detected."
        )


    # ======================================================
    # RUN DETAILED INVESTIGATION
    # ======================================================

    display_single_sample_analysis(
        selected_uploaded_sample,
        selected_actual_class
    )


    # ======================================================
    # EXPORT RESULTS
    # ======================================================

    st.divider()

    st.header(
        "Export Results"
    )


    export_results = (
        batch_results.copy()
    )


    export_results[
        "Failure Risk (%)"
    ] = export_results[
        "Failure Risk (%)"
    ].round(
        4
    )


    export_results[
        "Anomaly Score"
    ] = export_results[
        "Anomaly Score"
    ].round(
        6
    )


    csv_results = (
        export_results
        .drop(
            columns=[
                "Failure Risk"
            ]
        )
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )


    st.download_button(
        label=(
            "Download Analysis Results"
        ),
        data=csv_results,
        file_name=(
            "semiconductor_ai_results.csv"
        ),
        mime="text/csv"
    )


# ==========================================================
# MODEL INFORMATION
# ==========================================================

st.divider()

st.header(
    "Model Information"
)


info_col1, info_col2 = (
    st.columns(2)
)


with info_col1:

    st.subheader(
        "Yield Prediction"
    )

    st.write(
        "**Model:** Tuned Random Forest"
    )

    st.write(
        f"**Decision Threshold:** "
        f"{rf_threshold:.2f}"
    )

    st.write(
        "**Test FAIL Recall:** 52.4%"
    )

    st.write(
        "**Test FAIL Precision:** 19.6%"
    )

    st.write(
        "**Test ROC-AUC:** 0.769"
    )

    st.write(
        "**Purpose:** Predict whether a semiconductor "
        "process sample has elevated yield failure risk."
    )


with info_col2:

    st.subheader(
        "Process Anomaly Detection"
    )

    st.write(
        "**Model:** Isolation Forest"
    )

    st.write(
        f"**Anomaly Threshold:** "
        f"{anomaly_threshold:.6f}"
    )

    st.write(
        "**Test FAIL Recall:** 52.4%"
    )

    st.write(
        "**Test ROC-AUC:** 0.574"
    )

    st.write(
        "**Purpose:** Detect process behaviour that "
        "differs from learned normal PASS production."
    )


# ==========================================================
# DATASET INFORMATION
# ==========================================================

with st.expander(
    "Dataset & Method Information"
):

    st.write(
        "**Dataset:** UCI SECOM Semiconductor Manufacturing Dataset"
    )

    st.write(
        "**Original process variables:** 590"
    )

    st.write(
        "**Variables used after preprocessing:** 446"
    )

    st.write(
        f"**Random Forest threshold:** "
        f"{rf_threshold:.2f}"
    )

    st.write(
        f"**Isolation Forest threshold:** "
        f"{anomaly_threshold:.6f}"
    )

    st.write(
        "Random Forest provides supervised yield-risk "
        "prediction."
    )

    st.write(
        "Isolation Forest provides unsupervised "
        "process-anomaly detection."
    )

    st.write(
        "SHAP explains how individual process variables "
        "influence Random Forest predictions."
    )


# ==========================================================
# DISCLAIMER
# ==========================================================

st.divider()


st.caption(
    "Prototype developed using the UCI SECOM semiconductor "
    "manufacturing dataset. Process attributes are anonymized. "
    "AI predictions are experimental and intended for research "
    "and educational purposes. SHAP values explain model "
    "behaviour and do not establish causal relationships."

)

st.write("**Test FAIL Recall:** 52.4%")
st.write("**Test FAIL Precision:** 19.6%")
st.write("**Test ROC-AUC:** 0.769")