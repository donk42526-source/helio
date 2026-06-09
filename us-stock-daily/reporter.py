import json, os
from jinja2 import Environment, FileSystemLoader
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
_jinja = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
def generate_html(data: dict, output_path: str = "output/dashboard.html") -> str:
    stocks_js = []
    for s in data.get("stocks", []):
        sigs = s.get("signals",{})
        stocks_js.append({
            "ticker": s.get("ticker"), "sts": s.get("sts",0),
            "level": s.get("level"), "change_pct": s.get("change_pct",0),
            "alerts": s.get("alerts",[]),
            "signals": {"ma": {"score": sigs.get("ma",{}).get("score",0), "detail": sigs.get("ma",{}).get("detail","")},
                        "rsi": {"score": sigs.get("rsi",{}).get("score",0), "value": sigs.get("rsi",{}).get("value","")},
                        "macd": {"score": sigs.get("macd",{}).get("score",0), "detail": sigs.get("macd",{}).get("detail","")},
                        "volume": {"score": sigs.get("volume",{}).get("score",0), "detail": sigs.get("volume",{}).get("detail","")},
                        "bollinger": {"score": sigs.get("bollinger",{}).get("score",0), "detail": sigs.get("bollinger",{}).get("detail","")}}})
    tpl = _jinja.get_template("dashboard.html")
    html = tpl.render(date=data.get("date",""), mp=data.get("market_panic",{}),
                      stocks=data.get("stocks",[]), special_signals=data.get("special_signals",[]),
                      recommendation=data.get("recommendation",""), stocks_json=json.dumps(stocks_js))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path,"w",encoding="utf-8") as f: f.write(html)
    return output_path
