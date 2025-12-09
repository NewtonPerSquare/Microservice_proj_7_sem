from pathlib import Path
from datetime import datetime
import yaml
import requests

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go


CONFIG_PATH = Path(__file__).with_name("config.yaml")

DEFAULT_SETTINGS = {
    "web_master_url": "http://web_master:8000",
    "request_timeout": 60.0,
}


def load_config() -> dict:
    cfg = DEFAULT_SETTINGS.copy()
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                cfg.update(data)
    return cfg


settings = load_config()




def append_log(old_text, message: str) -> str:
    """добавить строку в лог с временной меткой"""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {message}"
    if not old_text:
        return line
    return str(old_text) + "\n" + line


def make_figure(y_true, y_pred, title: str) -> go.Figure:
    """gостроить график y_true vs y_pred"""
    fig = go.Figure()
    if y_true and y_pred:
        x = list(range(len(y_true)))
        fig.add_trace(go.Scatter(x=x, y=y_true, mode="lines", name="y_true"))
        fig.add_trace(go.Scatter(x=x, y=y_pred, mode="lines", name="y_pred"))
    fig.update_layout(
        title=title,
        xaxis_title="index",
        yaxis_title="value",
        template="plotly_white",
    )
    return fig


# ------------------ Dash-приложение ------------------

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    style={
        "fontFamily": 'system-ui, -apple-system, "Segoe UI", sans-serif',
        "margin": "16px",
        "backgroundColor": "#f5f5f5",
    },
    children=[
        html.H1("Soft Sensor Dashboard"),
        html.P("Интерактивный дашборд для сценариев Web Master и оценки soft sensor."),

        html.Div(
            style={
                "backgroundColor": "#ffffff",
                "borderRadius": "12px",
                "padding": "16px",
                "marginBottom": "16px",
                "boxShadow": "0 2px 8px rgba(0,0,0,0.05)",
            },
            children=[
                html.H2("Сервисы и сценарии"),

                html.Div(
                    style={"display": "flex", "gap": "8px", "marginBottom": "8px"},
                    children=[
                        html.Button(
                            "Запустить demo (x,y)",
                            id="btn-demo",
                            n_clicks=0,
                            style={"padding": "8px 12px"},
                        ),
                        html.Button(
                            "Обучить простую модель (x,y)",
                            id="btn-train-simple",
                            n_clicks=0,
                            style={"padding": "8px 12px"},
                        ),
                    ],
                ),

                html.Div(
                    style={"display": "flex", "gap": "8px", "marginBottom": "8px"},
                    children=[
                        html.Button(
                            "Обучить soft sensor (RandomForest)",
                            id="btn-train-soft-rf",
                            n_clicks=0,
                            style={"padding": "8px 12px"},
                        ),
                        html.Button(
                            "Обучить soft sensor (SDVAE)",
                            id="btn-train-soft-sd",
                            n_clicks=0,
                            style={"padding": "8px 12px"},
                        ),
                    ],
                ),

                html.Div(
                    style={"display": "flex", "gap": "8px", "alignItems": "center"},
                    children=[
                        html.Label("Число последних точек:"),
                        dcc.Input(
                            id="n-last",
                            type="number",
                            min=1,
                            max=100000,
                            value=100,
                            style={"width": "100px"},
                        ),
                        html.Button(
                            "Оценить RF soft sensor и построить график",
                            id="btn-eval-soft-rf",
                            n_clicks=0,
                            style={"padding": "8px 12px"},
                        ),
                        html.Button(
                            "Оценить SDVAE soft sensor и построить график",
                            id="btn-eval-soft-sd",
                            n_clicks=0,
                            style={"padding": "8px 12px"},
                        ),
                    ],
                ),
                html.Div(
                    style={"marginTop": "8px"},
                    children=[
                        html.A(
                            "Открыть HTML-отчёт RF soft sensor",
                            id="rf-report-link",
                            href="#",  #href будет задаваться колбэком
                            target="_blank",
                        )
                    ],
                ),

            ],
        ),

        html.Div(
            style={
                "backgroundColor": "#ffffff",
                "borderRadius": "12px",
                "padding": "16px",
                "marginBottom": "16px",
                "boxShadow": "0 2px 8px rgba(0,0,0,0.05)",
            },
            children=[
                html.H2("Логи"),
                html.Pre(
                    id="log",
                    style={
                        "height": "200px",
                        "overflowY": "auto",
                        "backgroundColor": "#111827",
                        "color": "#d1d5db",
                        "padding": "8px",
                        "borderRadius": "8px",
                        "fontFamily": "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace",
                        "fontSize": "13px",
                        "whiteSpace": "pre-wrap",
                    },
                ),
            ],
        ),

        html.Div(
            style={
                "backgroundColor": "#ffffff",
                "borderRadius": "12px",
                "padding": "16px",
                "marginBottom": "16px",
                "boxShadow": "0 2px 8px rgba(0,0,0,0.05)",
            },
            children=[
                html.H2("График y_true vs y_pred"),
                dcc.Graph(id="main-graph", figure=make_figure([], [], "")),
            ],
        ),
    ],
)



@app.callback(
    [Output("log", "children"), Output("main-graph", "figure")],
    [
        Input("btn-demo", "n_clicks"),
        Input("btn-train-simple", "n_clicks"),
        Input("btn-train-soft-rf", "n_clicks"),
        Input("btn-train-soft-sd", "n_clicks"),
        Input("btn-eval-soft-rf", "n_clicks"),
        Input("btn-eval-soft-sd", "n_clicks"),
    ],
    [
        State("n-last", "value"),
        State("log", "children"),
        State("main-graph", "figure"),
    ],
    prevent_initial_call=True,
)
def handle_buttons(
    n_demo,
    n_train_simple,
    n_train_rf,
    n_train_sd,
    n_eval_rf,
    n_eval_sd,
    n_last,
    log_text,
    current_fig,
):
    #кто нажал кнопку
    ctx = dash.callback_context
    if not ctx.triggered:
        fig = make_figure([], [], "") if current_fig is None else go.Figure(current_fig)
        return log_text, fig

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    #текущий график (если уже был)
    fig = make_figure([], [], "") if current_fig is None else go.Figure(current_fig)

    #нормализуем n_last
    try:
        n_last_int = int(n_last or 100)
        if n_last_int <= 0:
            n_last_int = 100
    except (TypeError, ValueError):
        n_last_int = 100

    if button_id == "btn-demo":
        log_text = append_log(log_text, "Запуск demo-сценария...")
        try:
            resp = requests.post(
                f"{settings['web_master_url']}/scenario/demo",
                json={"batch_size": 5},
                timeout=settings["request_timeout"],
            )
            resp.raise_for_status()
            data = resp.json()
            msg = (
                f"Demo: collected={data.get('collected')}, "
                f"stored_total={data.get('stored_total')}, "
                f"used_trained_model={data.get('used_trained_model')}"
            )
            log_text = append_log(log_text, msg)
        except Exception as e:
            log_text = append_log(log_text, f"Ошибка demo: {e}")
        return log_text, fig

    if button_id == "btn-train-simple":
        log_text = append_log(log_text, "Обучение простой модели (x,y)...")
        try:
            resp = requests.post(
                f"{settings['web_master_url']}/scenario/train_model",
                timeout=settings["request_timeout"],
            )
            resp.raise_for_status()
            data = resp.json()
            msg = (
                f"Train simple: rows_used={data.get('rows_used')}, "
                f"model_path={data.get('model_path')}"
            )
            log_text = append_log(log_text, msg)
        except Exception as e:
            log_text = append_log(log_text, f"Ошибка train simple: {e}")
        return log_text, fig

    if button_id == "btn-train-soft-rf":
        log_text = append_log(log_text, "Обучение RF soft sensor...")
        try:
            resp = requests.post(
                f"{settings['web_master_url']}/scenario/softsensor_train",
                timeout=settings["request_timeout"],
            )
            resp.raise_for_status()
            data = resp.json()
            msg = (
                f"RF train: rows_used={data.get('rows_used')}, "
                f"n_features={data.get('n_features')}, "
                f"mse={float(data.get('mse', 0.0)):.4f}, "
                f"r2={float(data.get('r2', 0.0)):.4f}"
            )
            log_text = append_log(log_text, msg)
        except Exception as e:
            log_text = append_log(log_text, f"Ошибка RF softsensor_train: {e}")
        return log_text, fig

    if button_id == "btn-train-soft-sd":
        log_text = append_log(log_text, "Обучение SDVAE soft sensor...")
        try:
            resp = requests.post(
                f"{settings['web_master_url']}/scenario/softsensor_sdvae_train",
                timeout=settings["request_timeout"],
            )
            resp.raise_for_status()
            data = resp.json()
            msg = (
                f"SDVAE train: rows_used={data.get('rows_used')}, "
                f"n_features={data.get('n_features')}, "
                f"mse={float(data.get('mse', 0.0)):.4f}, "
                f"r2={float(data.get('r2', 0.0)):.4f}"
            )
            log_text = append_log(log_text, msg)
        except Exception as e:
            log_text = append_log(log_text, f"Ошибка SDVAE softsensor_train: {e}")
        return log_text, fig

    if button_id == "btn-eval-soft-rf":
        log_text = append_log(
            log_text,
            f"Запрашиваю предсказания RF soft sensor для последних {n_last_int} точек...",
        )
        try:
            resp = requests.post(
                f"{settings['web_master_url']}/scenario/softsensor_eval",
                json={"n_last": n_last_int},
                timeout=settings["request_timeout"],
            )
            resp.raise_for_status()
            data = resp.json()
            y_true = data.get("y_true", [])
            y_pred = data.get("y_pred", [])
            log_text = append_log(
                log_text,
                f"RF eval: n_last={data.get('n_last')}, y_true={len(y_true)}, y_pred={len(y_pred)}",
            )
            if y_true and y_pred:
                fig = make_figure(y_true, y_pred, "RF soft sensor")
        except Exception as e:
            log_text = append_log(log_text, f"Ошибка RF softsensor_eval: {e}")
        return log_text, fig

    if button_id == "btn-eval-soft-sd":
        log_text = append_log(
            log_text,
            f"Запрашиваю предсказания SDVAE soft sensor для последних {n_last_int} точек...",
        )
        try:
            resp = requests.post(
                f"{settings['web_master_url']}/scenario/softsensor_sdvae_eval",
                json={"n_last": n_last_int},
                timeout=settings["request_timeout"],
            )
            resp.raise_for_status()
            data = resp.json()
            y_true = data.get("y_true", [])
            y_pred = data.get("y_pred", [])
            log_text = append_log(
                log_text,
                f"SDVAE eval: n_last={data.get('n_last')}, y_true={len(y_true)}, y_pred={len(y_pred)}",
            )
            if y_true and y_pred:
                fig = make_figure(y_true, y_pred, "SDVAE soft sensor")
        except Exception as e:
            log_text = append_log(log_text, f"Ошибка SDVAE softsensor_eval: {e}")
        return log_text, fig


    return log_text, fig

@app.callback(
    Output("rf-report-link", "href"),
    Input("n-last", "value"),
)
def update_report_link(n_last):
    #приводим n_last к адекватному виду
    try:
        n_last_int = int(n_last or 100)
        if n_last_int <= 0:
            n_last_int = 100
    except (TypeError, ValueError):
        n_last_int = 100

    #эта ссылка теперь будет использовать то же n_last, что и поле ввода
    return f"http://localhost:8000/report/softsensor?n_last={n_last_int}"


if __name__ == "__main__":
    #запуск Dash-сервера
    app.run(host="0.0.0.0", port=8000, debug=False)
