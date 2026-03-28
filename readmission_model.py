import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import plotly.graph_objects as go
from dash import Dash, html
from sklearn.preprocessing import LabelEncoder
import os
import dash_mantine_components as dmc
import plotly.express as px
from dash import Dash, html , dcc, Output, Input , ctx ,clientside_callback
import dash_bootstrap_components as dbc
import xgboost as xgb

app = Dash(__name__, external_stylesheets=[dbc.themes.LITERA], suppress_callback_exceptions=True)
server = app.server


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(BASE_DIR, "dataset", "train.csv"))


df.loc[df["max_glu_serum"].isna() , "max_glu_serum"] = "Not Measured"  #Imputed the nan values with not measured

df["medical_specialty"].replace("?", "Unknown", inplace=True)
df['payer_code'].replace('?', 'Unknown', inplace=True)
df['weight'].replace('?', 'Unknown', inplace=True)
df["race"].replace("?", "Unknown", inplace=True)
for col in ["diag_1", "diag_2", "diag_3"]:
    df[col].replace("?", "Unknown", inplace=True)

print((df=="?").sum())

TOTAL_ENCOUNTERS = df.shape[0]
print(TOTAL_ENCOUNTERS)

high_risk_readmitted_rate = round((df[df["readmitted"] == "<30"].shape[0]/TOTAL_ENCOUNTERS) * 100, 1)
print(high_risk_readmitted_rate)

moderate_risk_readmitted_rate = round((df[df["readmitted"] == ">30"].shape[0]/TOTAL_ENCOUNTERS) * 100, 1)
print(moderate_risk_readmitted_rate)

low_risk_readmitted_rate = round((df[df["readmitted"] == "NO"].shape[0]/TOTAL_ENCOUNTERS) * 100 , 1)
print(low_risk_readmitted_rate)

# fig = px.pie(df, names='readmitted', hole=.5, color= "readmitted", color_discrete_map=
#                                 {'NO':'#D3F527',
#                                  '<30':'#F54927',
#                                  '>30':'#F5B027'}, title="Readmission split" )
# fig.show()

# import plotly.express as px

# counts = df['readmitted'].value_counts()
# labels = counts.index
# values = counts.values

# # Create custom legend labels (like your image)
# total = values.sum()
# legend_labels = [
#     f"{label} days: {value/total:.1%}" if label != 'NO' 
#     else f"No readmit: {value/total:.1%}"
#     for label, value in zip(labels, values)
# ]

# fig = px.pie(
#     names=legend_labels,   # <-- legend text
#     values=values,
#     hole=0.5,
#     color=labels,
#     color_discrete_map={
#         'NO': 'green',
#         '<30': 'red',
#         '>30': 'orange'
#     }
# )

# fig.update_traces(
#     textinfo='none', 
#     marker=dict(line=dict(color='white', width=2))
# )

# fig.show()


# def create_pie_chart(pull=None, opacity=None):
#     counts = df['readmitted'].value_counts()
#     labels = counts.index
#     values = counts.values

#     total = values.sum()
#     legend_labels = [
#         f"{label} days: {value/total:.1%}" if label != 'NO'
#         else f"No readmit: {value/total:.1%}"
#         for label, value in zip(labels, values)
#     ]

#     fig = px.pie(
#         names=legend_labels,
#         values=values,
#         hole=0.5,
#         color=labels,
#         color_discrete_map={
#             'NO': "#4FA645",
#             '<30': '#e07a7a',
#             '>30': '#e6b566'
#         }
#     )

    
#     if pull is None:
#         pull = [0] * len(values)
#     if opacity is None:
#         opacity = [1] * len(values)

#     fig.update_traces(
#         textinfo='none',
#         pull=pull,
#         marker=dict(
#             line=dict(color='white', width=2),
#         ),
#         hovertemplate='%{label}<br>%{percent}<extra></extra>'
#     )

#     return fig
# fig = create_pie_chart()
# fig.show()


total_lt_thirty = (df["readmitted"] == "<30").sum()
total_gt_thirty = (df["readmitted"] == ">30").sum()
total_no_readmit = (df["readmitted"] == "NO").sum()
print(total_lt_thirty, total_gt_thirty, total_no_readmit)

readmission_rate = (
    df.groupby('age')['readmitted']
      .apply(lambda x: (x != 'NO').mean() * 100)
      .round(2)
      .reset_index(name='readmission_rate')
)


ages = readmission_rate["age"].apply(lambda x : x.split("-")[0].strip("["))
print(ages)
rates = readmission_rate["readmission_rate"]
def color_map(num):
    if num <= 41:
        return "#7aa6d1"
    elif num <= 46:
        return "#e6b566"
    else:
        return "#e07a7a"


age_group_bar_plot = px.bar (x=ages, y=rates, title='Readmission rate by age group')

colors = [color_map(rate) for rate in rates]


age_group_bar_plot.update_traces(
    marker_color=colors,
    text=[f"{r}%" for r in rates],
    textposition='outside'
)

age_group_bar_plot.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    title=dict(x=0),
    xaxis=dict(title=None),
    yaxis=dict(ticksuffix="%", gridcolor='#eee'),
    showlegend=False
    
)

# age_group_bar_plot.show()

changed_index=df["change"].value_counts().index
changed_count=df["change"].value_counts().count

readmitted_index=df["readmitted"].value_counts().index
readmitted_count=df["readmitted"].value_counts().count

# def medication_changeXOutput():
#     df_counts = pd.crosstab(df["change"], df["readmitted"])

#     df_counts = df_counts.rename(index={
#         "Ch": "Changed",
#         "No": "Not Change"
#     })

#     df_counts = df_counts.rename(columns={
#         "NO": "No readmit",
#         ">30": ">30 days",
#         "<30": "<30 days"
#     })

#     df_percent = df_counts.div(df_counts.sum(axis=1), axis=0) * 100
#     df_percent = df_percent.round(0).astype(int)

#     df_percent_str = df_percent.astype(str) + "%"

#     colors = [
#         ["#dff0d8"]*len(df_percent), 
#         ["#fce8d6"]*len(df_percent),  
#         ["#f7dada"]*len(df_percent)   
#     ]

#     fig = go.Figure(data=[go.Table(
#         header=dict(
#             values=["Medication change"] + list(df_percent.columns),
#             fill_color='white',
#             align='center'
#         ),
#         cells=dict(
#             values=[df_percent.index] + [df_percent_str[col] for col in df_percent.columns],
#             fill_color=[["#ffffff"]*len(df_percent)] + colors,  
#             align='center'
#         )
#     )])

#     fig.update_layout(
#         title="Medication Change X Outcome"
#     )
#     return fig

# fig=medication_changeXOutput()
# fig.show()


days_outcome = df.groupby("readmitted")["time_in_hospital"].mean()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

saved_clf = xgb.XGBClassifier()
saved_clf.load_model(os.path.join(BASE_DIR, "xgboost_model.json"))

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(["<30", ">30", "NO"])




#PLOTS

def create_card(title, value, subtitle, percent, color):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, style={"fontSize": "12px", "color": "gray", "fontWeight": "600"}),
            
            html.H4(value, style={"color": color, "marginTop": "5px"}),
            
            html.Div(subtitle, style={"fontSize": "12px", "color": "gray"}),
            
            
            dbc.Progress(
                value=percent,
                color=color,
                style={"height": "2px", "marginTop": "10px"},
            )
        ]),
        style={
            "borderRadius": "12px",
            "boxShadow": "0 2px 6px rgba(0,0,0,0.05)",
        }
    )


def create_pie_chart(pull=None, opacity=None):
    counts = df['readmitted'].value_counts()
    labels = counts.index
    values = counts.values

    total = values.sum()
    legend_labels = [
        f"{label} days: {value/total:.1%}" if label != 'NO'
        else f"No readmit: {value/total:.1%}"
        for label, value in zip(labels, values)
    ]

    fig = px.pie(
        names=legend_labels,
        values=values,
        hole=0.5,
        color=labels,
        color_discrete_map={
            'NO': "#4FA645",
            '<30': '#e07a7a',
            '>30': '#e6b566'
        },title="Readmission split"
    )

    
    if pull is None:
        pull = [0] * len(values)
    if opacity is None:
        opacity = [1] * len(values)

    fig.update_traces(
        textinfo='none',
        pull=pull,
        marker=dict(
            line=dict(color='white', width=2),
        ),
        hovertemplate='%{label}<br>%{percent}<extra></extra>'
    )

    return fig

def plot_age_group_bar_plot(ages, rates, title):
    age_group_bar_plot = px.bar(x=ages, y=rates, title=title)
    
    colors = [color_map(rate) for rate in rates]
    
    age_group_bar_plot.update_traces(
        marker_color=colors,
        text=[f"{r}%" for r in rates],
        textposition='outside',
        marker_line_width=0, 
        hovertemplate='<b>%{x}</b><br>Rate: %{y}%<extra></extra>',
        selector=dict(type='bar')
    )
    
  
    age_group_bar_plot.update_traces(
        marker_line_width=3,
        hoverinfo='y',
    )

    age_group_bar_plot.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        title=dict(x=0),
        xaxis=dict(title=None),
        yaxis=dict(title=None, ticksuffix="%", gridcolor='#eee'),  
        showlegend=False,
        hovermode='x',  
    )
    
    return age_group_bar_plot


def race_distribution(opacity=None):
    labels = df["race"].value_counts().index
    values = df["race"].value_counts().values

    fig = px.bar(
        x=labels,
        y=values,
        color=labels,
        color_discrete_map=
        {
        'Hispanic': "#4FA645",        
        'Other': '#e07a7a', 
        'Unknown': '#e6b566',        
        'Caucasian': "#7aa6d1",         
        'AfricanAmerican': "#b59ad6",            
        "Asian": "#5cc0b3"            
    
        }
    )
    
    if opacity is None:
        opacity = [1] * len(values)

    fig.update_traces(
        marker=dict(
            line=dict(color='white', width=2),
        ),

        hovertemplate=
        "<b>Race:</b> %{x}<br>" +
        "<b>Count:</b> %{y}<extra></extra>"
        )
    
    fig.update_layout(
        title="Race Distribution",
        xaxis_title=None,
        yaxis_title=None,
        legend_title=None,
        plot_bgcolor='white',
        showlegend=False
    )

    fig.update_xaxes(tickangle=-45)

    return fig


def medication_changeXOutput():
    df_counts = pd.crosstab(df["change"], df["readmitted"])

    df_counts = df_counts.rename(index={
        "Ch": "Changed",
        "No": "Not Change"
    })

    df_counts = df_counts.rename(columns={
        "NO": "No readmit",
        ">30": ">30 days",
        "<30": "<30 days"
    })

    df_percent = df_counts.div(df_counts.sum(axis=1), axis=0) * 100
    df_percent = df_percent.round(0).astype(int)

    df_percent_str = df_percent.astype(str) + "%"

    colors = [
        ["#dff0d8"]*len(df_percent), 
        ["#fce8d6"]*len(df_percent),  
        ["#f7dada"]*len(df_percent)   
    ]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Medication"] + list(df_percent.columns),
            fill_color='white',
            align='center'
        ),
        cells=dict(
            values=[df_percent.index] + [df_percent_str[col] for col in df_percent.columns],
            fill_color=[["#ffffff"]*len(df_percent)] + colors,  
            align='center',
            height=50,
            font=dict(size=12),
        )
    )])

    fig.update_layout(
        title="Medication Change X Outcome",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def avg_hostpital_days(opacity=None):    
    days_outcome = df.groupby("readmitted")["time_in_hospital"].mean()
    categories = days_outcome.index.tolist()
    mean_days = days_outcome.values.tolist()

    days=[]
    for i in range(5):
        days.append(f'{i+1}d')

    colors={
        "NO":"#7aa6d1",
        ">30": "#e6b566",
        "<30":"#e07a7a"
    }

    fig = px.bar(
        df,
        y=categories,
        x=mean_days,
        orientation='h',
        color=categories,
        color_discrete_map=colors     
    )

    fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis_title=None,
    yaxis_title=None,
    legend_title=None,
    showlegend=False,
    title="Avg hospital days by outcome",
)
    
    fig.update_traces(
         hovertemplate=
        "<b>Avg days:</b> %{x}" 
    )
    
    fig.update_xaxes(
        tickvals=[1,2,3,4,5],
        ticktext=days
    )
    
    if opacity is None:
        opacity = [1] * len(categories)

    return fig




#Helper functions 
# ── Prediction function ───────────────────────────────────────────────────────
def get_model_prediction(inpatient_visits, emergency_visits, medications_num,
                         hosp_time, diagnoses_num, lab_procdrs_num,
                         insulin_lvl, A1C_lvl):
    categorical_cols = ["insulin", "A1Cresult"]
    sample_data = {
        "number_inpatient":   [int(inpatient_visits)],
        "number_emergency":   [int(emergency_visits)],
        "num_medications":    [int(medications_num)],
        "time_in_hospital":   [int(hosp_time)],
        "number_diagnoses":   [int(diagnoses_num)],
        "num_lab_procedures": [int(lab_procdrs_num)],
        "insulin":            [str(insulin_lvl)],
        "A1Cresult":          [str(A1C_lvl)],
    }
    sample_df = pd.DataFrame(sample_data)
    for col in categorical_cols:
        sample_df[col] = sample_df[col].astype("category")

    prediction     = saved_clf.predict(sample_df)
    props = saved_clf.predict_proba(sample_df)[0]
    original_label = label_encoder.inverse_transform(prediction)
    prop = max(props)  # confidence of the predicted class
    pred = original_label[0]


    if pred == "<30":
        color = "#E07A7A"
        val   = np.random.randint(70,100)
    elif pred == ">30":
        color = "#E6B566"
        val   = np.random.randint(20,69)
    else:
        color = "#4FA645"
        val   = np.random.randint(0,19)  

    return color, val, pred


# BROWN    = "#8B5E1A"
# BG_RIGHT = "#EDE6D8"
# TEXT     = "#2C2416"
# SOFT     = "#9C8E77"
# BG_MAIN  = "#F7F4EE"
BORDER   = "#C9B99A"

DEFAULT_SLIDER_COLOR = "#4FA645"

# All numeric slider IDs – drives both layout and callback outputs
SLIDER_IDS = ["inpatient", "emergency", "meds", "time", "diagnoses", "labs"]


def dmc_slider(label, sid, min_, max_, value):
    return html.Div([
        html.Div(label, style={"fontSize": "13px", "marginBottom": "4px"}),
        dmc.Slider(
            id=sid,
            min=min_,
            max=max_,
            step=1,
            value=value,
            color=DEFAULT_SLIDER_COLOR,   # updated live via Output(sid, "color")
            showLabelOnHover=True,
            size="sm",
            styles={
                "track": {"cursor": "pointer"},
                "thumb": {"borderWidth": "2px"},
            },
        ),
    ], style={"marginBottom": "14px"})


def thermometer(val, pred_label, color):
    pct = int(val)

    if pred_label == "<30":
        risk_text, css_class, pill_class = "High risk",     "thermo-high",     "pill-high"
    elif pred_label == ">30":
        risk_text, css_class, pill_class = "Moderate risk", "thermo-moderate", "pill-moderate"
    else:
        risk_text, css_class, pill_class = "Low risk",      "thermo-low",      "pill-low"

    ticks = html.Div([
        html.Div(f"{v}%", style={"fontSize": "11px"})
        for v in [80, 60, 40, 20, 0]
    ], style={
        "display": "flex", "flexDirection": "column",
        "justifyContent": "space-between",
        "height": "90%", "paddingBottom": "26px",
        "marginRight": "10px", "textAlign": "right",
    })

    thermo_col = html.Div([
        html.Div(className="thermo-bar", children=[
            html.Div(className="thermo-fill",
                     style={"height": f"{max(pct, 4)}%", "background": color})
        ]),
        html.Div(className="thermo-bulb", style={"background": color}),
    ], style={"display": "flex", "flexDirection": "column",
              "alignItems": "center", "height": "90%"})

    info_col = html.Div([
        html.Div("Predicted <30 day readmission",
                 style={"fontSize": "13px", "marginBottom": "8px"}),
        html.Div(f"{pct}%",
                 style={"fontSize": "64px", "fontWeight": "700",
                        "lineHeight": "1"}),
        html.Div(f"Predicted: {pred_label}",
                 style={"fontSize": "13px", "marginTop": "4px"}),
        html.Div(risk_text, className=f"pill {pill_class}"),
    ], style={"paddingLeft": "32px", "display": "flex",
              "flexDirection": "column", "justifyContent": "center"})

    return html.Div(
        [ticks, thermo_col, info_col],
        className=css_class,
        style={"display": "flex", "flexDirection": "row",
               "alignItems": "stretch", "height": "100%", "width": "100%"},
    )


# # ── Sidebar ───────────────────────────────────────────────────────────────────
# sidebar = html.Div([
#     html.Div([
#         html.Div("DiabetesIQ",
#                  style={"fontWeight": "700", "fontSize": "18px", "color": TEXT}),
#         html.Div("readmission analytics",
#                  style={"fontSize": "11px", "color": SOFT, "marginTop": "2px"}),
#     ], style={"marginBottom": "10px"}),
#     html.Div("DASHBOARDS", className="sidebar-label"),
#     html.Div([html.Span("▪ "), html.Span("Risk predictor")],
#              className="sidebar-item active"),
# ], className="sidebar")




#PAGES

def dashboard():
    return dbc.Container([
    
    dbc.Row([
        dbc.Col("Dashboard", 
            style={
            "fontSize": "28px",
            "fontWeight": "bold",
            "fontFamily": "Arial",
            # "marginTop":"10px",
            "marginButtom":"20px"
            # "textAlign": "center"
        })
    ]),

    dbc.Row([
        dbc.Col(create_card("TOTAL ENCOUNTERS", f"{TOTAL_ENCOUNTERS:,}", "across 10 years", 100, "primary")),
        dbc.Col(create_card("READMITTED <30 DAYS", f"{high_risk_readmitted_rate}%", f"{total_lt_thirty:,} high-risk", high_risk_readmitted_rate, "danger")),
        dbc.Col(create_card("READMITTED >30 DAYS", f"{moderate_risk_readmitted_rate}%", f"{total_gt_thirty:,} patients", moderate_risk_readmitted_rate, "warning")),
        dbc.Col(create_card("NO READMISSION", f"{low_risk_readmitted_rate}%", f"{total_no_readmit:,} patients", low_risk_readmitted_rate, "success")),
    ], className="g-3", style={"marginBottom": "30px"}),

    dbc.Row(
        [
        dbc.Col(
            dcc.Graph(
                figure=plot_age_group_bar_plot(ages, rates, "Readmission Rate by Age Group"),
                style={"height": "400px"}
        )),

        dbc.Col(
            dcc.Graph(
                id="pie-chart", 
                figure=create_pie_chart()))
            ]),


    dbc.Row(
        [
            dbc.Col(
                dcc.Graph(
                    id="race_distribution",
                    figure=race_distribution(),
                    style={"height": "400px"}
                ),
                width=4,  
            ),

            dbc.Col(
                dcc.Graph(
                    id="crosstable",
                    figure=medication_changeXOutput(),
                    style={"height": "400px"}  
                ),
                width=4,  
            ),

            dbc.Col(
                dcc.Graph(
                    id="avg_hostpital_days",
                    figure=avg_hostpital_days(),
                    style={"height": "400px"} 
                ),
                width=4,
            )
        ],
        align="start",  
        className="g-3", 
    )
    ]

    )


@app.callback(
    Output("pie-chart", "figure"),
    Input("pie-chart", "hoverData")
)
def update_pie_on_hover(hoverData):
    counts = df['readmitted'].value_counts()
    n = len(counts)

    pull = [0] * n
    opacity = [0.3] * n  

    if hoverData:
        idx = hoverData['points'][0]['pointNumber']
        pull[idx] = 0.1        
        opacity[idx] = 1      
    else:
        opacity = [1] * n    

    return create_pie_chart(pull, opacity)


#Infrence Page

def inference_page():
    return dmc.MantineProvider([html.Div([

    # Store: carries prediction color between the two callbacks
    dcc.Store(id="pred-color-store", data=DEFAULT_SLIDER_COLOR),

    # Invisible anchor for clientside_callback output
    html.Div(id="slider-color-style", style={"display": "none"}),

    html.Div([
        html.Div([
            # Header
            html.Div([
                html.Div("Risk predictor",   style={
            "fontSize": "28px",
            "fontWeight": "bold",
            "fontFamily": "Arial",
            # "marginTop":"10px",
            "marginButtom":"20px"
            # "textAlign": "center"
        }),
            ], style={
                # "display": "flex", "alignItems": "flex-start",
                # "justifyContent": "center",
                # "padding": "18px 28px 14px 28px",
                "borderBottom": f"1px solid {BORDER}"
                # , "flexShrink": "0",
            }),

            html.Div([

                # LEFT – sliders
                html.Div([
                    html.Div("Patient inputs", style={
                        "fontSize": "20px", "fontWeight": "700",
                        "marginBottom": "14px",
                    }),
                    dmc_slider("Prior inpatient visits",  "inpatient",  0, 10,  2),
                    dmc_slider("Emergency visits",        "emergency",  0, 30,  0),
                    dmc_slider("Medications count",       "meds",       0, 30, 14),
                    dmc_slider("Time in hospital (days)", "time",       1, 14,  4),
                    dmc_slider("Number of diagnoses",     "diagnoses",  0, 10,  6),
                    dmc_slider("Lab procedures",          "labs",       0, 50, 19),

                    html.Div([
                        html.Div([
                            html.Div("Insulin",
                                     style={"fontSize": "13px",
                                            "marginBottom": "4px"}),
                            dcc.Dropdown(["No", "Up", "Down", "Steady"],
                                         value="No", id="insulin", clearable=False),
                        ], style={"flex": "1"}),
                        html.Div([
                            html.Div("A1C result",
                                     style={"fontSize": "13px", 
                                            "marginBottom": "4px"}),
                            dcc.Dropdown(["None", "Norm", ">7", ">8"],
                                         value="None", id="a1c", clearable=False),
                        ], style={"flex": "1"}),
                    ], style={"display": "flex", "gap": "20px", "marginTop": "6px"}),

                ], style={"width": "48%", "padding": "22px 28px", "flexShrink": "0"}),

                # RIGHT – thermometer
                html.Div(id="thermo", style={
                    "flex": "1",
                    "borderLeft": f"1px solid {BORDER}",
                    "padding": "28px 36px", "display": "flex",
                    "alignItems": "stretch",
                }),

            ], style={"display": "flex", "flex": "1",
                      "overflow": "hidden", "minHeight": "0"}),

        ], style={"flex": "1", "display": "flex", "flexDirection": "column",
                  "overflow": "hidden", "minHeight": "0"}),

    ], style={
        "display": "flex", "height": "100vh", "overflow": "hidden",
        "fontFamily": "Arial" , "background":"#ffffff"
      
    }),
], style={"height": "100vh", "overflow": "hidden"})])


# ── Shared inputs ─────────────────────────────────────────────────────────────
_INPUTS = [
    Input("inpatient", "value"),
    Input("emergency", "value"),
    Input("meds",      "value"),
    Input("time",      "value"),
    Input("diagnoses", "value"),
    Input("labs",      "value"),
    Input("insulin",   "value"),
    Input("a1c",       "value"),
]


# ── Callback 1 – thermometer + store + one color Output per slider ────────────
# Exactly mirrors the reference pattern: Output("slider", "color") per slider.
@app.callback(
    Output("thermo",           "children"),
    Output("pred-color-store", "data"),
    *[Output(sid, "color") for sid in SLIDER_IDS],
    *_INPUTS,
)
def update_dashboard(inpatient, emergency, meds, time, diagnoses, labs, insulin, a1c):
    color, val, pred_label = get_model_prediction(
        inpatient, emergency, meds, time, diagnoses, labs, insulin, a1c
    )
    thermo_component = thermometer(val, pred_label, color)
    # thermometer  +  store color  +  same color pushed to each slider
    return (thermo_component, color) + (color,) * len(SLIDER_IDS)


# ── Callback 2 – clientside safety net: keeps <style> in sync too ─────────────
# Handles any edge case where DMC re-renders and the prop momentarily resets.
clientside_callback(
    """
    function(color) {
        if (!color) color = "#4FA645";
        var id  = "dmc-slider-dynamic";
        var el  = document.getElementById(id);
        if (!el) {
            el    = document.createElement("style");
            el.id = id;
            document.head.appendChild(el);
        }
        el.textContent = [
            ".mantine-Slider-bar   { background-color: " + color + " !important; }",
            ".mantine-Slider-thumb { border-color: "     + color + " !important; background-color: " + color + " !important; }",
            ".mantine-Slider-label { background-color: " + color + " !important; }"
        ].join(" ");
        return color;
    }
    """,
    Output("slider-color-style", "children"),
    Input("pred-color-store",    "data"),
)



PAGES = {"dashboard": "Dashboard", "inference": "Inference"}
NAV_ICONS = {"dashboard": "▦", "inference": "⟳"}

app.layout = html.Div([
    dcc.Store(id="active-page", data="dashboard"),

    html.Div([
        # Left sidebar
        html.Div([
            html.Div("Readmissions Model", style={
                "fontWeight": "500", "fontSize": "15px",
                "padding": "1.25rem 1rem", "borderBottom": "1px solid #dee2e6",
                "marginBottom": "0.5rem"
            }),
            *[
                html.Button(
                    [html.Span(NAV_ICONS[key], style={"marginRight": "8px", "fontSize": "14px"}), label],
                    id=f"nav-{key}",
                    n_clicks=0,
                    style={
                        "display": "flex", "alignItems": "center",
                        "width": "calc(100% - 16px)", "margin": "0 8px 4px",
                        "border": "none", "borderRadius": "6px",
                        "padding": "10px 12px", "cursor": "pointer",
                        "fontSize": "13px", "textAlign": "left",
                        "background": "transparent", "color": "#212529"
                    }
                )
                for key, label in PAGES.items()
            ]
        ], style={
            "width": "180px", "minWidth": "180px",
            "borderRight": "1px solid #dee2e6",
            "background": "#f8f9fa", "height": "100vh",
        }),

        # Main content
        html.Div(id="page-content", style={
            "flex": "1", "padding": "2rem",
            "overflowY": "auto", "background": "#ffffff", "height": "100vh"
        }),
    ], style={"display": "flex", "height": "100vh"})
])

# Switch page
@app.callback(
    Output("active-page", "data"),
    [Input(f"nav-{key}", "n_clicks") for key in PAGES],
    prevent_initial_call=True
)
def set_page(*_):
    triggered = ctx.triggered_id
    return triggered.replace("nav-", "") if triggered else "dashboard"

# Render selected page
@app.callback(
    Output("page-content", "children"),
    Input("active-page", "data")
)
def render_page(page):
    if page == "dashboard":
        return dashboard()  
    else :
        return inference_page()

# Highlight active nav button 
@app.callback(
    [Output(f"nav-{key}", "style") for key in PAGES],
    Input("active-page", "data")
)
def highlight_nav(active):
    base = {
        "display": "flex", "alignItems": "center",
        "width": "calc(100% - 16px)", "margin": "0 8px 4px",
        "border": "none", "borderRadius": "6px",
        "padding": "10px 12px", "cursor": "pointer",
        "fontSize": "13px", "textAlign": "left"
    }
    return [
        {**base, "background": "#e8f0fe", "color": "#0d6efd", "fontWeight": "500"}
        if key == active else
        {**base, "background": "transparent", "color": "#212529"}
        for key in PAGES
    ]


    
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",                 # bind to all network interfaces
        port=int(os.environ.get("PORT", 8050)),  # use Render's assigned port
        debug=True
    )
